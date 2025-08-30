# Zer0-Day Bokeh: Cybernetic Depth Field Adjuster

This script applies a depth-of-field (bokeh) effect to an image using a pre-trained deep learning model to generate a depth map. Your vision, sharpened.

## Features

*   **AI-Powered Depth Mapping:** Uses the MiDaS/DPT depth estimation models from Intel to create realistic depth maps.
*   **Multiple Focus Modes:**
    *   **Automatic:** Focuses on the center of the image by default.
    *   **Percentile-based:** Automatically focuses on a specific depth percentile.
    *   **Explicit:** Manually set a normalized focus depth.
    *   **Interactive:** Click on a preview window to set the exact focal point (requires a graphical environment).
*   **Customizable Bokeh:**
    *   Adjust the number of aperture blades to create polygonal bokeh shapes (or a circular one).
    *   Control the rotation of the polygonal aperture.
    *   Fine-tune sharpness and the in-focus band.
*   **Layered Blurring:** Generates a more realistic and smoother blur effect by blending multiple blurred layers.

## Dependencies

The script requires the following Python packages:

*   `torch`
*   `torchvision`
*   `torchaudio`
*   `opencv-contrib-python`
*   `numpy`
*   `Pillow`
*   `timm`

## Installation

1.  Clone the repository or download the script.
2.  Install the required packages using pip:
    ```bash
    pip install torch torchvision torchaudio opencv-contrib-python numpy Pillow timm
    ```
3.  **Note:** The first time you run the script, it will download the pre-trained MiDaS model (approx. 1.3 GB for the default model). This requires an active internet connection.

## Usage

Run the script from your terminal. The only required argument is `--input`.

### Basic Usage

This will process the input image and save the result in the `output/` directory.

```bash
python dof_bokeh1.py --input IMG_2037.jpeg
```

### Advanced Options

You can combine various arguments to customize the output.

**Specify a different depth model:** (Smaller models are faster but less accurate)

```bash
python dof_bokeh1.py --input IMG_2037.jpeg --model_type MiDaS_small
```

**Use auto-focus based on a depth percentile:** (Focus on the furthest 25% of the scene)

```bash
python dof_bokeh1.py --input IMG_2037.jpeg --focus_percentile 0.75
```

**Create a hexagonal bokeh effect:**

```bash
python dof_bokeh1.py --input IMG_2037.jpeg --blades 6
```

**Enable the interactive focus preview:** (A window will pop up. Click to set focus, then press any key.)

```bash
python dof_bokeh1.py --input IMG_2037.jpeg --preview
```

### All Arguments

```
usage: dof_bokeh1.py [-h] --input INPUT [--outdir OUTDIR] [--blades BLADES] [--angle ANGLE] [--max_radius MAX_RADIUS] [--sharpness SHARPNESS] [--band BAND] [--mask_feather MASK_FEATHER]
                     [--guided_mask] [--layers LAYERS] [--layer_blur_scale LAYER_BLUR_SCALE] [--preview] [--focus_percentile FOCUS_PERCENTILE] [--focus FOCUS] [--bins BINS]
                     [--invert_depth] [--model_type MODEL_TYPE]

Cyberpunk Image Depth of Field Adjuster

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to the input image.
  --outdir OUTDIR       Directory to save the output image.
  --blades BLADES       Number of aperture blades for bokeh shape (0 for circular). 8 is octagonal.
  --angle ANGLE         Rotation angle for polygonal bokeh.
  --max_radius MAX_RADIUS
                        Maximum blur radius. If omitted and --preview is used a preview-optimized default is applied.
  --sharpness SHARPNESS
                        Sharpness of the focus transition.
  --band BAND           Width of in-focus region (larger = more in focus). If omitted and --preview is used a preview-optimized default is applied.
  --mask_feather MASK_FEATHER
                        Radius for mask feathering (reduces halo/ghosting).
  --guided_mask         Use guided filter for mask feathering (if available).
  --layers LAYERS       Number of discrete depth layers for layered blur. If omitted and --preview is used a preview-optimized default is applied.
  --layer_blur_scale LAYER_BLUR_SCALE
                        Per-layer blur scale multiplier. If omitted and --preview is used a preview-optimized default is applied.
  --preview             Enable interactive click-to-focus preview.
  --focus_percentile FOCUS_PERCENTILE
                        Set focus automatically at a depth percentile.
  --focus FOCUS         Set focus explicitly at a normalized depth value (0.0-1.0).
  --bins BINS           Number of bins for some depth models.
  --invert_depth        Invert depth polarity if your model outputs inverse depth (near=larger values).
  --model_type MODEL_TYPE
                        Depth model type: DPT_Large, DPT_Hybrid, or MiDaS_small.
```
