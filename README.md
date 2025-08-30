# Zer0-Day Bokeh: Cybernetic Depth Field Adjuster

This project provides tools to apply a depth-of-field (bokeh) effect to an image using a pre-trained deep learning model to generate a depth map. Your vision, sharpened.

It offers two ways to use the effect: a user-friendly GUI and a command-line interface.

## Features

*   **AI-Powered Depth Mapping:** Uses the MiDaS/DPT depth estimation models from Intel to create realistic depth maps.
*   **Customizable Bokeh:** Adjust the number of aperture blades, rotation, sharpness, and more.
*   **Layered Blurring:** Generates a more realistic and smoother blur effect.
*   **Interactive and CLI Control:** Choose between a graphical interface for interactive focus selection or a command-line tool for batch processing.

## Dependencies

The script requires the following Python packages:

*   `torch`
*   `torchvision`
*   `torchaudio`
*   `opencv-contrib-python`
*   `numpy`
*   `Pillow`
*   `timm`
*   `tkinter` (usually included with Python)

## Installation

1.  Clone the repository or download the script.
2.  Install the required packages using pip:
    ```bash
    pip install torch torchvision torchaudio opencv-contrib-python numpy Pillow timm
    ```
3.  **Note:** The first time you run the script, it will download the pre-trained MiDaS model (approx. 1.3 GB for the default model). This requires an active internet connection.

---

## 1. GUI Application

The GUI provides an interactive way to apply the bokeh effect. You can load an image, click to set the focus point, adjust settings with sliders, and see the results instantly.

### Usage

Run the `app.py` script to launch the application:

```bash
python app.py
```

### Controls

*   **File Menu:** Open and save image files.
*   **Image Preview:** Click on the image to set the exact focal point.
*   **Model:** Select the depth estimation model (larger models are more accurate but slower).
*   **Sliders:**
    *   **Blades:** Number of aperture blades for bokeh shape (0 for circular).
    *   **Angle:** Rotation angle for polygonal bokeh.
    *   **Max Radius:** Maximum blur radius.
    *   **Sharpness:** Sharpness of the focus transition.
    *   **Band:** Width of the in-focus region.
    *   **Mask Feather:** Radius for mask feathering to reduce halos.
    *   **Layers:** Number of discrete depth layers for a smoother blur.
    *   **Layer Blur Scale:** Per-layer blur scale multiplier.
*   **Checkboxes:**
    *   **Invert Depth:** Invert the depth map if needed.
    *   **Guided Mask:** Use a guided filter for more accurate mask feathering.
*   **Process Image Button:** Applies the settings to the image.

---

## 2. Command-Line Interface (CLI)

The CLI tool (`dof_bokeh1.py`) is suitable for batch processing or integration into scripts.

### Basic Usage

This will process the input image and save the result in the `output/` directory.

```bash
python dof_bokeh1.py --input IMG_2037.jpeg
```

### All Arguments

Here are the available command-line arguments for `dof_bokeh1.py`:

```
usage: dof_bokeh1.py [-h] --input INPUT [--outdir OUTDIR] [--blades BLADES] [--angle ANGLE] [--max_radius MAX_RADIUS]
                     [--sharpness SHARPNESS] [--band BAND] [--mask_feather MASK_FEATHER] [--guided_mask]
                     [--layers LAYERS] [--layer_blur_scale LAYER_BLUR_SCALE] [--focus_x FOCUS_X]
                     [--focus_y FOCUS_Y] [--invert_depth] [--model_type MODEL_TYPE]

Cyberpunk Image Depth of Field Adjuster

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to the input image.
  --outdir OUTDIR       Directory to save the output image.
  --blades BLADES       Number of aperture blades for bokeh shape (0 for circular).
  --angle ANGLE         Rotation angle for polygonal bokeh.
  --max_radius MAX_RADIUS
                        Maximum blur radius.
  --sharpness SHARPNESS
                        Sharpness of the focus transition.
  --band BAND           Width of in-focus region.
  --mask_feather MASK_FEATHER
                        Radius for mask feathering (reduces halo/ghosting).
  --guided_mask         Use guided filter for mask feathering.
  --layers LAYERS       Number of discrete depth layers for layered blur.
  --layer_blur_scale LAYER_BLUR_SCALE
                        Per-layer blur scale multiplier.
  --focus_x FOCUS_X     X coordinate of focus point (0-1).
  --focus_y FOCUS_Y     Y coordinate of focus point (0-1).
  --invert_depth        Invert depth polarity if your model outputs inverse depth.
  --model_type MODEL_TYPE
                        Depth model type: DPT_Large, DPT_Hybrid, or MiDaS_small.
```
