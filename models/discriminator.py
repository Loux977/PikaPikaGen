import torch
import torch.nn as nn
import math

from models.blocks import ResBlockD

class Discriminator(nn.Module):
    def __init__(self, text_dim=256, base_channels=32, image_size=128, use_spectral_norm=False, down_type="stride_conv"):
        super().__init__()

        # Compute number of layers from image size (until 4x4)
        assert image_size in [64, 128, 256], "Only 64, 128 or 256 resolution supported"
        num_layers = int(math.log2(image_size // 4))

        self.base = DiscriminatorBase(base_channels, num_layers, use_spectral_norm, down_type)
        self.cond_head = ConditionalHead(in_channels=self.base.out_channels, text_dim=text_dim, base_channels=base_channels)

    def forward(self, x, text_emb):
        features = self.base(x)
        out = self.cond_head(features, text_emb)
        return out


class DiscriminatorBase(nn.Module):
    def __init__(self, base_channels=32, num_layers=6, use_spectral_norm=False, down_type="stride_conv"):
        super().__init__()

        # Initial conv to map 3-channel RGB input to base_channels
        self.initial_conv = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)

        # Compute channel progression
        channel_list = [base_channels * min(2 ** i, 8) for i in range(num_layers)]
        # Example (base_channels=32, num_blocks=6): [32, 64, 128, 256, 256, 256]

        blocks = []
        channels = base_channels
        # Create layers
        for next_channels in channel_list:
            blocks.append(ResBlockD(channels, next_channels, use_spectral_norm, down_type))
            channels = next_channels

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = channels

    def forward(self, x):
        """
        x: [B, 3, H, W]
        """
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        return x


class ConditionalHead(nn.Module):
    def __init__(self, in_channels, text_dim=256, base_channels=32):
        super().__init__()

        self.joint_conv = nn.Sequential(
            nn.Conv2d(in_channels + text_dim, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, features, text_emb):
        """
        features: [B, C, H', W']
        text_emb: [B, text_dim]
        """
        B, _, H, W = features.shape

        # Expand text embedding to spatial map
        text_map = text_emb.view(B, -1, 1, 1)
        text_map = text_map.repeat(1, 1, H, W) # [B, text_dim, H', W']

        # Concatenate
        fused = torch.cat([features, text_map], dim=1) # [B, C + text_dim, H', W']

        # Joint conv after text fusion (DF-GAN style)
        out = self.joint_conv(fused)  # [B, 1, 1, 1]
        out = out.view(B) # [B]
        return out
