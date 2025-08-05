import torch
import torch.nn as nn
import math

from models.blocks import ResBlockG, CrossAttentionBlock, Upsample

class Generator(nn.Module):
    def __init__(self, z_dim=128, text_dim=256, base_channels=32, image_size=128, up_type="nearest"): 
        super().__init__()
        self.z_dim = z_dim
        self.text_dim = text_dim

        # Compute number of layers based on target image resolution (start from 4x4)
        assert image_size in [64, 128, 256], "Only 64, 128 or 256 resolution supported"
        num_layers = int(math.log2(image_size // 4))  # 4x4 base

        # Compute channels per block
        channel_list = [base_channels * min(2 ** i, 8) for i in range(num_layers)]
        channel_list = channel_list[::-1]  # reverse
        # Example (base_channels=32, image_size=256): [256, 256, 256, 128, 64, 32]

        self.channel_list = channel_list

        # Initial projection uses first (largest) channels
        self.init_proj = nn.Linear(self.z_dim + self.text_dim, 4 * 4 * channel_list[0])

        blocks = []
        channels = channel_list[0]
        # Create Layers: (i) Upsample -> (ii) ResBlock -> (iii) CrossAttention
        for next_channels in channel_list:
            blocks.append(Upsample(channels, up_type))
            blocks.append(ResBlockG(channels, next_channels))
            blocks.append(CrossAttentionBlock(next_channels, text_dim))
            channels = next_channels

        self.blocks = nn.ModuleList(blocks)

        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, per_token_emb, global_emb):
        """
        z: [B, z_dim]
        per_token_text_emb: [B, N_tokens, text_dim]
        global_emb: [B, text_dim]
        """
        B = z.size(0)

        conditioned_input = torch.cat((z, global_emb), dim=1)

        x = self.init_proj(conditioned_input)
        x = x.view(B, self.channel_list[0], 4, 4)

        for block in self.blocks:
            if isinstance(block, CrossAttentionBlock):
                x = block(x, per_token_emb)
            else:
                x = block(x)

        x = self.to_rgb(x)
        return x
