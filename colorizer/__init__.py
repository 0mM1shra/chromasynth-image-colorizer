"""
colorizer — Deep Image Colorization Package
============================================
Production-grade implementation of the ECCV 2016 automatic colorization
network (Zhang et al.). Provides the ECCVGenerator model with pretrained
weights and Lab color-space preprocessing/postprocessing utilities.

Usage:
    from colorizer import eccv16, preprocess_img, postprocess_tens

    model = eccv16(pretrained=True)  # Auto-downloads weights from S3
    model.eval().to('cuda')

    tens_orig_l, tens_rs_l = preprocess_img(rgb_numpy_array)
    out_ab = model(tens_rs_l.to('cuda'))
    rgb_out = postprocess_tens(tens_orig_l, out_ab.cpu())
"""

from .eccv16 import ECCVGenerator, eccv16
from .util import load_img, resize_img, preprocess_img, postprocess_tens
from .base_color import BaseColor

__all__ = [
    "BaseColor",
    "ECCVGenerator",
    "eccv16",
    "load_img",
    "resize_img",
    "preprocess_img",
    "postprocess_tens",
]
