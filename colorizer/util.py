"""
util.py — Image Processing Utilities
=====================================
Preprocessing and postprocessing functions for the Lab-based colorization
pipeline. Handles RGB↔Lab conversions, tensor reshaping, and resolution
management between original and model-native 256×256 spatial dimensions.
"""

from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F


def load_img(img_path: str) -> np.ndarray:
    """Load an image file and ensure it is 3-channel RGB.

    Args:
        img_path: Absolute or relative path to the image file.

    Returns:
        NumPy array of shape [H, W, 3] in uint8 RGB format.
    """
    out_np = np.asarray(Image.open(img_path))
    if out_np.ndim == 2:
        # Grayscale → replicate across 3 channels
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np


def resize_img(img: np.ndarray, HW: tuple = (256, 256), resample: int = 3) -> np.ndarray:
    """Resize image to target (H, W) using PIL resampling.

    Args:
        img:      Input numpy array [H, W, C].
        HW:       Target (height, width) tuple.
        resample: PIL resampling filter (3 = BICUBIC).

    Returns:
        Resized numpy array.
    """
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))


def preprocess_img(
    img_rgb_orig: np.ndarray,
    HW: tuple = (256, 256),
    resample: int = 3,
) -> tuple:
    """Convert RGB image to Lab and extract L channel tensors.

    Produces two tensors:
        1. tens_orig_l: L channel at original resolution [1, 1, H_orig, W_orig]
        2. tens_rs_l:   L channel resized to model input  [1, 1, 256, 256]

    The original-resolution tensor is preserved for postprocessing
    (bilinear upsampling the predicted ab channels back to original size).

    Args:
        img_rgb_orig: RGB numpy array [H, W, 3] in uint8.
        HW:           Target resize dimensions for model input.
        resample:     PIL resampling filter index.

    Returns:
        Tuple of (tens_orig_l, tens_rs_l) as float32 torch.Tensors.
    """
    # Resize the RGB image to model's native resolution
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    # Convert both resolutions to CIELAB color space
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    # Extract the L (lightness) channel — shape [H, W]
    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]

    # Reshape to [1, 1, H, W] batch tensor format
    tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
    tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]

    return (tens_orig_l, tens_rs_l)


def postprocess_tens(
    tens_orig_l: torch.Tensor,
    out_ab: torch.Tensor,
    mode: str = "bilinear",
) -> np.ndarray:
    """Reconstruct full RGB image from L tensor and predicted ab tensor.

    Pipeline:
        1. Upsample predicted ab from model resolution to original resolution
        2. Concatenate L + ab to form full Lab image
        3. Convert Lab → RGB via skimage

    Args:
        tens_orig_l: Original-resolution L tensor [1, 1, H_orig, W_orig].
        out_ab:      Predicted ab tensor [1, 2, H_model, W_model].
        mode:        Interpolation mode for upsampling.

    Returns:
        RGB numpy array [H_orig, W_orig, 3] in float64, range [0, 1].
    """
    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # Bilinear upsample ab channels to match original spatial dimensions
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode="bilinear")
    else:
        out_ab_orig = out_ab

    # Concatenate L + ab along channel dim → [1, 3, H, W] Lab tensor
    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)

    # Convert to numpy [H, W, 3] and transform Lab → RGB
    lab_np = out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0))
    rgb_np = color.lab2rgb(lab_np)

    return rgb_np
