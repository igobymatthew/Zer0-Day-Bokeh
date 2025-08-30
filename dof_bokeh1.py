import argparse
import cv2
import numpy as np
import torch
import argparse
import cv2
import numpy as np
import torch
import os
from PIL import Image

# Zer0-Day Bokeh: Cybernetic Depth Field Adjuster
#
# Your vision, sharpened.

def load_depth_model(device, model_type="DPT_Large"):
    """
    Loads the MiDaS/DPT depth estimation model and its transforms.
    Returns a tuple (model, transform).
    model_type: one of 'DPT_Large', 'DPT_Hybrid', 'MiDaS_small'
    """
    print(f"Loading MiDaS model ({model_type})...")
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
        model.to(device)
        model.eval()

        # Load the transforms module and pick the correct transform
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "MiDaS_small":
            transform = midas_transforms.small_transform
        else:
            # DPT_Large and DPT_Hybrid use the dpt_transform
            transform = midas_transforms.dpt_transform

        return model, transform
    except Exception as e:
        print(f"Error loading MiDaS model ({model_type}): {e}")
        return None, None

def generate_depth_map(model, transform, image, device):
    """
    Generates a depth map from the input image using the MiDaS model.
    """
    # MiDaS requires specific preprocessing
    transform_input = transform(image).to(device)
    with torch.no_grad():
        prediction = model(transform_input)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()

    # Normalize depth map to 0-1 range for consistency
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    return depth_map_normalized


# Global variable to store click data for the preview window
click_data = {'x': -1, 'y': -1}

def mouse_callback(event, x, y, flags, param):
    """CV2 mouse callback function to capture click coordinates."""
    if event == cv2.EVENT_LBUTTONDOWN:
        click_data['x'] = x
        click_data['y'] = y
        print(f"Focus point selected at (x={x}, y={y})")

def get_focal_depth(args, depth_map):
    """
    Determines the focal depth based on user arguments.
    """
    if args.preview:
        print("Interactive preview enabled. Click on the image to set focus. Press any key to continue.")
        preview_window_name = "Zer0-Day Bokeh: Click to Focus"

        # Create a copy of the depth map for display, as a colorized version
        depth_display = (depth_map * 255).astype(np.uint8)
        depth_display_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)

        cv2.namedWindow(preview_window_name)
        cv2.setMouseCallback(preview_window_name, mouse_callback)

        while True:
            cv2.imshow(preview_window_name, depth_display_color)
            key = cv2.waitKey(1) & 0xFF
            # Exit on any key press, after a click has been registered
            if key != 255 and click_data['x'] != -1:
                break

        cv2.destroyWindow(preview_window_name)

        # Use the clicked point to set the focal depth
        return depth_map[click_data['y'], click_data['x']]

    elif args.focus_percentile is not None:
        # Auto-focus at a certain percentile of the depth map
        focal_depth = np.percentile(depth_map, args.focus_percentile * 100)
        return focal_depth

    elif args.focus is not None:
        # Use a direct, normalized value
        return args.focus

    else:
        # Default to the center of the image if no focus method is specified
        h, w = depth_map.shape
        return depth_map[h // 2, w // 2]


def generate_bokeh_kernel(radius, blades, angle_rad):
    """Generates a bokeh kernel, circular or polygonal."""
    kernel = np.zeros((radius * 2 + 1, radius * 2 + 1))
    center = radius

    if blades == 0:  # Circular bokeh
        cv2.circle(kernel, (center, center), radius, (1, 1, 1), -1)
    else:  # Polygonal bokeh
        points = []
        for i in range(blades):
            theta = (2 * np.pi * i / blades) + angle_rad
            x = int(center + radius * np.cos(theta))
            y = int(center + radius * np.sin(theta))
            points.append((x, y))
        points = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(kernel, points, (1, 1, 1))

    # Normalize the kernel
    kernel /= np.sum(kernel)
    return kernel

def normalize_depth(d):
    """
    Normalize any raw depth map into [0,1] float32. Robust to constant maps.
    """
    d = d.astype(np.float32)
    d_min = d.min()
    d = d - d_min
    rng = d.max() - d.min()
    if rng < 1e-6:
        return np.zeros_like(d, dtype=np.float32)
    return d / rng


def compute_focus_mask(depth_norm, focus_depth, band=0.05):
    """
    depth_norm: [0..1] float32
    focus_depth: scalar in [0..1]
    band: width of in-focus region (larger = more in focus)
    Returns mask in [0..1], 1 = KEEP SHARP, 0 = BLUR
    """
    sigma = max(band, 1e-6)
    mask = np.exp(-0.5 * ((depth_norm - focus_depth) / sigma) ** 2)
    return np.clip(mask, 0.0, 1.0)


def apply_dof_bokeh(img_bgr, depth_raw, click_xy, band=0.05, blur_sigma: float = 9.0, invert_depth: bool = False, layers: int = 1, layer_blur_scale: float = 1.0):
    """
    Single-pass or layered DOF composite.
    If layers>1, creates a discrete set of blurred images and blends them using depth-dependent weights.
    Returns: out_bgr(uint8), depth_norm, focus_depth, mask
    """
    # 1) Normalize depth
    d = normalize_depth(depth_raw)

    # 2) Optional polarity correction
    if invert_depth:
        d = 1.0 - d

    # 3) Get focus depth at click
    x, y = click_xy
    h, w = d.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    focus_depth = float(d[y, x])

    # Diagnostic: print clicked depth raw and normalized to detect polarity
    try:
        raw_val = float(depth_raw[y, x])
    except Exception:
        raw_val = None
    print(f"Clicked depth (raw, norm): {raw_val}, {focus_depth}")

    # Compute a simple in-focus mask for diagnostics / optional feathering
    mask = compute_focus_mask(d, focus_depth, band=band)  # HxW in [0,1]

    # If layered blur requested, build discrete blur levels and blend
    if layers is None:
        layers = 1
    layers = max(1, int(layers))

    if layers <= 1:
        # Existing single-blur approach, optionally feather mask
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

    # Layered approach
    L = layers
    # coc: distance from focus plane in [0,1]
    coc = np.clip(np.abs(d - focus_depth), 0.0, 1.0)

    # level centers (0..1) and sigma for soft blending between adjacent levels
    centers = np.linspace(0.0, 1.0, L).astype(np.float32)
    level_sigma = max(0.01, 1.0 / float(L) * 0.75)

    # compute weights: shape HxWxL
    # expand coc to HxWx1 then compute gaussian weights around centers
    coc_exp = coc[:, :, None]
    diff = (coc_exp - centers[None, None, :]) / level_sigma
    weights = np.exp(-0.5 * (diff ** 2))
    weights_sum = np.sum(weights, axis=2, keepdims=True)
    weights = weights / (weights_sum + 1e-8)

    # precompute blurred images for each level
    blurred_levels = []
    for i in range(L):
        if L == 1:
            sigma_i = blur_sigma * layer_blur_scale
        else:
            sigma_i = blur_sigma * layer_blur_scale * (float(i) / float(L - 1))
        if sigma_i <= 0.5:
            # effectively no blur => use original
            blurred_i = img_bgr.astype(np.float32)
        else:
            blurred_i = cv2.GaussianBlur(img_bgr, ksize=(0, 0), sigmaX=sigma_i, sigmaY=sigma_i, borderType=cv2.BORDER_REPLICATE).astype(np.float32)
        blurred_levels.append(blurred_i)

    # composite: weighted sum across levels
    out_f = np.zeros_like(blurred_levels[0], dtype=np.float32)
    for i in range(L):
        w = weights[:, :, i][..., None]
        out_f += w * blurred_levels[i]

    out = np.clip(out_f, 0, 255).astype(np.uint8)
    return out, d, focus_depth, mask

# Global mask options
_mask_options = {'mask_feather': None, 'guided_mask': False}

def _set_mask_options(band, mask_feather, guided_mask):
    _mask_options['mask_feather'] = float(mask_feather) if mask_feather is not None else None
    _mask_options['guided_mask'] = bool(guided_mask)

# Replace previous iterative apply_bokeh_effect with a thin wrapper around apply_dof_bokeh
def apply_bokeh_effect(image, depth_map, focal_depth, args):
    """
    Wrapper that prepares parameters and calls the robust DOF helper.
    """
    # image is BGR uint8 as loaded by cv2
    img_bgr = image

    # Decide click coordinate: prefer interactive click if used, otherwise center
    h, w = depth_map.shape[:2]
    if args.preview and click_data.get('x', -1) != -1:
        click_xy = (click_data['x'], click_data['y'])
    else:
        click_xy = (w // 2, h // 2)

    # Decide defaults; when preview mode is active and the user did NOT supply
    # specific values we use the preview-tuned defaults requested by the user.
    # Determine effective max_radius
    if getattr(args, 'max_radius', None) is not None:
        eff_max_radius = int(args.max_radius)
    else:
        eff_max_radius = 12 if getattr(args, 'preview', False) else 28

    # Determine effective band
    if getattr(args, 'band', None) is not None:
        band = float(args.band)
    else:
        band = 0.12 if getattr(args, 'preview', False) else float(np.clip(0.4 * (1.0 - float(getattr(args, 'sharpness', 12)) / 100.0) + 0.02, 0.02, 0.5))

    # Determine layering defaults
    if getattr(args, 'layers', None) is not None:
        layers_val = int(args.layers)
    else:
        layers_val = 6 if getattr(args, 'preview', False) else 1

    if getattr(args, 'layer_blur_scale', None) is not None:
        layer_blur_scale = float(args.layer_blur_scale)
    else:
        layer_blur_scale = 1.2 if getattr(args, 'preview', False) else 1.0

    blur_sigma = max(1.0, float(eff_max_radius) / 3.0)

    # Pass mask feathering options into the DOF helper via function attributes
    _set_mask_options(band, getattr(args, 'mask_feather', None), getattr(args, 'guided_mask', False))

    out, depth_norm, focus_depth_used, mask = apply_dof_bokeh(
        img_bgr=img_bgr,
        depth_raw=depth_map,
        click_xy=click_xy,
        band=band,
        blur_sigma=blur_sigma,
        invert_depth=bool(getattr(args, 'invert_depth', False)),
        layers=layers_val,
        layer_blur_scale=layer_blur_scale,
    )

    return out


def main(args):
    """
    Main function to orchestrate the depth-of-field effect.
    """
    print("Initializing Zer0-Day Bokeh...")
    print(f"Input image: {args.input}")
    print("Backend: MiDaS")

    # --- 1. Load Image ---
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not read image from {args.input}")
        return

    print(f"Image loaded successfully. Shape: {image.shape}")

    # --- 2. Load Depth Model ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    depth_model, transform = load_depth_model(DEVICE, args.model_type)
    if depth_model is None:
        return

    # --- 3. Get Depth Map ---
    depth_map = generate_depth_map(depth_model, transform, image, DEVICE)
    if depth_map is None:
        print("Error: Could not generate depth map.")
        return

    print(f"Depth map generated. Shape: {depth_map.shape}, Min: {depth_map.min()}, Max: {depth_map.max()}")

    # --- 4. Handle Focus ---
    focal_depth = get_focal_depth(args, depth_map)
    print(f"Focal depth set to: {focal_depth:.4f}")

    # --- 5. Apply Bokeh Effect ---
    output_image = apply_bokeh_effect(image, depth_map, focal_depth, args)
    print("Bokeh effect applied.")

    # --- 6. Save Output ---
    basename = os.path.splitext(os.path.basename(args.input))[0]
    output_filename = f"{basename}_bokeh.jpg"
    output_path = os.path.join(args.outdir, output_filename)

    cv2.imwrite(output_path, output_image)
    print(f"Output image saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cyberpunk Image Depth of Field Adjuster")

    parser.add_argument('--input', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--outdir', type=str, default='output', help='Directory to save the output image.')
    # Bokeh settings
    parser.add_argument('--blades', type=int, default=8, help='Number of aperture blades for bokeh shape (0 for circular). 8 is octagonal.')
    parser.add_argument('--angle', type=int, default=0, help='Rotation angle for polygonal bokeh.')
    parser.add_argument('--max_radius', type=int, default=None, help='Maximum blur radius. If omitted and --preview is used a preview-optimized default is applied.')
    parser.add_argument('--sharpness', type=int, default=12, help='Sharpness of the focus transition.')
    parser.add_argument('--band', type=float, help='Width of in-focus region (larger = more in focus). If omitted and --preview is used a preview-optimized default is applied.')
    parser.add_argument('--mask_feather', type=float, help='Radius for mask feathering (reduces halo/ghosting).')
    parser.add_argument('--guided_mask', action='store_true', help='Use guided filter for mask feathering (if available).')
    parser.add_argument('--layers', type=int, default=None, help='Number of discrete depth layers for layered blur. If omitted and --preview is used a preview-optimized default is applied.')
    parser.add_argument('--layer_blur_scale', type=float, default=None, help='Per-layer blur scale multiplier. If omitted and --preview is used a preview-optimized default is applied.')

    # Focus settings
    parser.add_argument('--preview', action='store_true', help='Enable interactive click-to-focus preview.')
    parser.add_argument('--focus_percentile', type=float, help='Set focus automatically at a depth percentile.')
    parser.add_argument('--focus', type=float, help='Set focus explicitly at a normalized depth value (0.0-1.0).')

    # Model-specific settings (like --bins for Depth Anything)
    parser.add_argument('--bins', type=int, default=10, help='Number of bins for some depth models.')
    parser.add_argument('--invert_depth', action='store_true', help='Invert depth polarity if your model outputs inverse depth (near=larger values).')
    parser.add_argument('--model_type', type=str, default='DPT_Large', help='Depth model type: DPT_Large, DPT_Hybrid, or MiDaS_small.')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    main(args)
