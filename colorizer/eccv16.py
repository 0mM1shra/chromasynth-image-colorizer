"""
eccv16.py — ECCV 2016 Colorization Generator
=============================================
Complete PyTorch implementation of the automatic image colorization network
from "Colorful Image Colorization" (Zhang et al., ECCV 2016).

Architecture: 8-block fully-convolutional encoder-decoder
    Blocks 1–3: Strided convolutions for spatial downsampling (1→64→128→256)
    Blocks 4–7: Dilated convolutions at 512 channels for receptive field expansion
    Block 8:    Transposed convolution upsampling + softmax classification
                over 313 ab quantized bins → 1×1 conv → 2-channel ab regression
                → 4× bilinear upsample to input resolution

Input:  [B, 1, 256, 256]  — L channel (lightness)
Output: [B, 2, 256, 256]  — ab channels (chrominance), unnormalized

Pre-trained weights are auto-downloaded from Amazon S3 via torch.utils.model_zoo.
"""

import torch
import torch.nn as nn
import numpy as np

from .base_color import BaseColor


class ECCVGenerator(BaseColor):
    """ECCV 2016 Automatic Colorization Generator.

    An 8-block encoder-decoder CNN that maps a single-channel Lightness
    tensor to a 2-channel ab chrominance prediction through 313-bin
    softmax classification followed by regression decoding.
    """

    def __init__(self, norm_layer: type = nn.BatchNorm2d) -> None:
        super(ECCVGenerator, self).__init__()

        # ─── Block 1: 1→64 channels, stride-2 downsample (256→128) ───
        model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        model1 += [nn.ReLU(True)]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True)]
        model1 += [nn.ReLU(True)]
        model1 += [norm_layer(64)]

        # ─── Block 2: 64→128 channels, stride-2 downsample (128→64) ──
        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
        model2 += [nn.ReLU(True)]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True)]
        model2 += [nn.ReLU(True)]
        model2 += [norm_layer(128)]

        # ─── Block 3: 128→256 channels, stride-2 downsample (64→32) ──
        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [norm_layer(256)]

        # ─── Block 4: 256→512 channels, stride-1 (spatial 32×32) ─────
        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [norm_layer(512)]

        # ─── Block 5: 512→512, dilation=2 (effective RF expansion) ───
        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [norm_layer(512)]

        # ─── Block 6: 512→512, dilation=2 ────────────────────────────
        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model6 += [nn.ReLU(True)]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model6 += [nn.ReLU(True)]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model6 += [nn.ReLU(True)]
        model6 += [norm_layer(512)]

        # ─── Block 7: 512→512, stride-1 (no dilation) ────────────────
        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [norm_layer(512)]

        # ─── Block 8: 512→256 upsample + 313-class softmax head ─────
        model8 = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
        model8 += [nn.ReLU(True)]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model8 += [nn.ReLU(True)]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model8 += [nn.ReLU(True)]
        # Final classification layer into 313 quantized ab bins
        model8 += [nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True)]

        # Register all blocks as nn.Sequential submodules
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        # Softmax over the 313-class dimension for probability distribution
        self.softmax = nn.Softmax(dim=1)

        # Regression head: 313 probability bins → 2-channel ab output
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)

        # 4× bilinear upsampling to recover spatial resolution (32→128→original via postprocess)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, input_l: torch.Tensor) -> torch.Tensor:
        """Forward pass: L channel → predicted ab channels.

        Args:
            input_l: Lightness tensor of shape [B, 1, H, W] in [0, 100] range.

        Returns:
            Unnormalized ab tensor of shape [B, 2, H, W].
        """
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)

        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))


def eccv16(pretrained: bool = True) -> ECCVGenerator:
    """Factory function to create and optionally load pretrained ECCVGenerator.

    Args:
        pretrained: If True, downloads weights from Amazon S3 (~130MB).

    Returns:
        ECCVGenerator model instance.
    """
    model = ECCVGenerator()
    if pretrained:
        import torch.utils.model_zoo as model_zoo

        model.load_state_dict(
            model_zoo.load_url(
                "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
                map_location="cpu",
                check_hash=True,
            )
        )
    return model
