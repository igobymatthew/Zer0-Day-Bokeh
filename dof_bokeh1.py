import argparse
import cv2
import numpy as np
import torch
import os
from PIL import Image

class BokehProcessor:
    def __init__(self, model_type="DPT_Large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.transform = self._load_depth_model(model_type)
        self.depth_map = None

    def _load_depth_model(self, model_type):
        print(f"Loading MiDaS model ({model_type})...")
        try:
            model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
            model.to(self.device)
            model.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if model_type == "MiDaS_small":
                transform = midas_transforms.small_transform
            else:
                transform = midas_transforms.dpt_transform
            return model, transform
        except Exception as e:
            print(f"Error loading MiDaS model ({model_type}): {e}")
            return None, None

    def generate_depth_map(self, image):
        if self.model is None or self.transform is None:
            print("Depth model not loaded.")
            return None
        transform_input = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.model(transform_input)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        self.depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        return self.depth_map

    def process_image(self, image, focus_point, blades, angle, max_radius, sharpness, band, mask_feather, guided_mask, layers, layer_blur_scale, invert_depth):
        if self.depth_map is None:
            print("Please generate depth map first.")
            return None

        h, w = self.depth_map.shape[:2]
        click_xy = (int(focus_point[0] * w), int(focus_point[1] * h))

        eff_max_radius = max_radius if max_radius is not None else 28

        if band is None:
            band = float(np.clip(0.4 * (1.0 - float(sharpness) / 100.0) + 0.02, 0.02, 0.5))

        layers_val = layers if layers is not None else 1
        layer_blur_scale_val = layer_blur_scale if layer_blur_scale is not None else 1.0
        blur_sigma = max(1.0, float(eff_max_radius) / 3.0)

        _set_mask_options(band, mask_feather, guided_mask)

        out, _, _, _ = apply_dof_bokeh(
            img_bgr=image,
            depth_raw=self.depth_map,
            click_xy=click_xy,
            band=band,
            blur_sigma=blur_sigma,
            invert_depth=bool(invert_depth),
            layers=layers_val,
            layer_blur_scale=layer_blur_scale_val,
        )
        return out

def normalize_depth(d):
    d = d.astype(np.float32)
    d_min = d.min()
    d = d - d_min
    rng = d.max() - d.min()
    if rng < 1e-6:
        return np.zeros_like(d, dtype=np.float32)
    return d / rng

def compute_focus_mask(depth_norm, focus_depth, band=0.05):
    sigma = max(band, 1e-6)
    mask = np.exp(-0.5 * ((depth_norm - focus_depth) / sigma) ** 2)
    return np.clip(mask, 0.0, 1.0)

_mask_options = {'mask_feather': None, 'guided_mask': False}

def _set_mask_options(band, mask_feather, guided_mask):
    _mask_options['mask_feather'] = float(mask_feather) if mask_feather is not None else None
    _mask_options['guided_mask'] = bool(guided_mask)

def apply_dof_bokeh(img_bgr, depth_raw, click_xy, band=0.05, blur_sigma: float = 9.0, invert_depth: bool = False, layers: int = 1, layer_blur_scale: float = 1.0):
    d = normalize_depth(depth_raw)
    if invert_depth:
        d = 1.0 - d

    x, y = click_xy
    h, w = d.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    focus_depth = float(d[y, x])

    mask = compute_focus_mask(d, focus_depth, band=band)

    if layers <= 1:
        mask_f = mask.astype(np.float32)
        mf = _mask_options.get('mask_feather', None)
        guided = _mask_options.get('guided_mask', False)
        if mf is not None and mf > 0:
            try:
                if guided:
                    import cv2.ximgproc as xip
                    guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    radius = max(1, int(mf))
                    eps = max(1e-6, (mf * 0.01) ** 2)
                    mask_f = xip.guidedFilter(guide, mask_f, radius, eps)
                else:
                    k = max(1, int(mf) * 2 + 1)
                    mask_f = cv2.GaussianBlur(mask_f, (k, k), sigmaX=mf, sigmaY=mf, borderType=cv2.BORDER_REPLICATE)
            except Exception:
                k = max(1, int(mf) * 2 + 1)
                mask_f = cv2.GaussianBlur(mask_f, (k, k), sigmaX=mf, sigmaY=mf, borderType=cv2.BORDER_REPLICATE)
        mask3 = mask_f[..., None]
        blurred = cv2.GaussianBlur(img_bgr, ksize=(0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma, borderType=cv2.BORDER_REPLICATE)
        out = (mask3 * img_bgr.astype(np.float32) + (1.0 - mask3) * blurred.astype(np.float32))
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out, d, focus_depth, mask

    L = layers
    coc = np.clip(np.abs(d - focus_depth), 0.0, 1.0)
    centers = np.linspace(0.0, 1.0, L).astype(np.float32)
    level_sigma = max(0.01, 1.0 / float(L) * 0.75)
    coc_exp = coc[:, :, None]
    diff = (coc_exp - centers[None, None, :]) / level_sigma
    weights = np.exp(-0.5 * (diff ** 2))
    weights_sum = np.sum(weights, axis=2, keepdims=True)
    weights = weights / (weights_sum + 1e-8)
    blurred_levels = []
    for i in range(L):
        if L == 1:
            sigma_i = blur_sigma * layer_blur_scale
        else:
            sigma_i = blur_sigma * layer_blur_scale * (float(i) / float(L - 1))
        if sigma_i <= 0.5:
            blurred_i = img_bgr.astype(np.float32)
        else:
            blurred_i = cv2.GaussianBlur(img_bgr, ksize=(0, 0), sigmaX=sigma_i, sigmaY=sigma_i, borderType=cv2.BORDER_REPLICATE).astype(np.float32)
        blurred_levels.append(blurred_i)
    out_f = np.zeros_like(blurred_levels[0], dtype=np.float32)
    for i in range(L):
        w = weights[:, :, i][..., None]
        out_f += w * blurred_levels[i]
    out = np.clip(out_f, 0, 255).astype(np.uint8)
    return out, d, focus_depth, mask

def main_cli():
    parser = argparse.ArgumentParser(description="Cyberpunk Image Depth of Field Adjuster")
    parser.add_argument('--input', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--outdir', type=str, default='output', help='Directory to save the output image.')
    parser.add_argument('--blades', type=int, default=8, help='Number of aperture blades for bokeh shape (0 for circular).')
    parser.add_argument('--angle', type=int, default=0, help='Rotation angle for polygonal bokeh.')
    parser.add_argument('--max_radius', type=int, default=None, help='Maximum blur radius.')
    parser.add_argument('--sharpness', type=int, default=12, help='Sharpness of the focus transition.')
    parser.add_argument('--band', type=float, default=None, help='Width of in-focus region.')
    parser.add_argument('--mask_feather', type=float, default=None, help='Radius for mask feathering.')
    parser.add_argument('--guided_mask', action='store_true', help='Use guided filter for mask feathering.')
    parser.add_argument('--layers', type=int, default=None, help='Number of discrete depth layers.')
    parser.add_argument('--layer_blur_scale', type=float, default=None, help='Per-layer blur scale multiplier.')
    parser.add_argument('--focus_x', type=float, default=0.5, help='X coordinate of focus point (0-1).')
    parser.add_argument('--focus_y', type=float, default=0.5, help='Y coordinate of focus point (0-1).')
    parser.add_argument('--invert_depth', action='store_true', help='Invert depth polarity.')
    parser.add_argument('--model_type', type=str, default='DPT_Large', help='Depth model type.')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not read image from {args.input}")
        return

    processor = BokehProcessor(model_type=args.model_type)
    processor.generate_depth_map(image)
    output_image = processor.process_image(
        image,
        (args.focus_x, args.focus_y),
        args.blades,
        args.angle,
        args.max_radius,
        args.sharpness,
        args.band,
        args.mask_feather,
        args.guided_mask,
        args.layers,
        args.layer_blur_scale,
        args.invert_depth
    )

    basename = os.path.splitext(os.path.basename(args.input))[0]
    output_filename = f"{basename}_bokeh.jpg"
    output_path = os.path.join(args.outdir, output_filename)
    cv2.imwrite(output_path, output_image)
    print(f"Output image saved to: {output_path}")

if __name__ == '__main__':
    main_cli()
