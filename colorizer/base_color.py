"""
base_color.py — BaseColor Module
=================================
Foundation nn.Module providing CIELAB color-space normalization constants
and helper methods inherited by all generator architectures.

Normalization scheme (from Zhang et al., ECCV 2016):
    L channel:  normalized as  (L - 50) / 100   →  range ~ [-0.5, 0.5]
    ab channels: normalized as  ab / 110         →  range ~ [-1.0, 1.0]
"""

import torch
from torch import nn


class BaseColor(nn.Module):
    """Base class for colorization generators.

    Attributes:
        l_cent:  Lightness centering constant (50.0).
        l_norm:  Lightness normalization divisor (100.0).
        ab_norm: Chrominance normalization divisor (110.0).
    """

    def __init__(self) -> None:
        super(BaseColor, self).__init__()
        self.l_cent: float = 50.0
        self.l_norm: float = 100.0
        self.ab_norm: float = 110.0

    def normalize_l(self, in_l: torch.Tensor) -> torch.Tensor:
        """Center and scale L channel to ~ [-0.5, 0.5]."""
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l: torch.Tensor) -> torch.Tensor:
        """Reverse L normalization back to [0, 100] range."""
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab: torch.Tensor) -> torch.Tensor:
        """Scale ab channels to ~ [-1.0, 1.0]."""
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab: torch.Tensor) -> torch.Tensor:
        """Reverse ab normalization back to [-110, 110] range."""
        return in_ab * self.ab_norm
