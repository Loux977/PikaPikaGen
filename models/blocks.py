import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlockG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            #nn.Dropout(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        #
        if in_channels == out_channels:
            self.skip_proj = nn.Identity()
        else:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip_proj(x)

        out = self.in_layers(x)
        out = self.out_layers(out)

        return out + identity

"""
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, text_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Group normalization before projecting queries
        self.norm = nn.GroupNorm(num_groups=32, num_channels=hidden_dim)

        # Linear projections for query, key, value are handled by MultiheadAttention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(text_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(text_dim, hidden_dim, bias=False)

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Final output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, text_emb):
        B, C, H, W = x.shape

        # Group norm on x
        x_norm = self.norm(x)

        # Flatten spatial dims: [B, C, H, W] → [B, HW, C]
        x_flat = x_norm.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

        # Project queries, keys, values
        q = self.q_proj(x_flat)          # [B, HW, C]
        k = self.k_proj(text_emb)        # [B, N, C]
        v = self.v_proj(text_emb)        # [B, N, C]

        # Normalize Q and K before passing to MultiheadAttention (cosine attention)
        eps = 1e-6
        q = q / (q.norm(dim=-1, keepdim=True) + eps)
        k = k / (k.norm(dim=-1, keepdim=True) + eps)

        # Cross attention: query = image features, key & value = text embeddings
        attn_out, _ = self.attn(q, k, v)  # [B, HW, C]

        # Output projection
        out = self.out_proj(attn_out)     # [B, HW, C]

        # Reshape back to [B, C, H, W]
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # Residual connection
        return x + out
"""

# SDPA Cross Attention     
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, text_dim, num_heads=4):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads # Calculate head_dim

        self.norm = nn.GroupNorm(num_groups=32, num_channels=hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(text_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(text_dim, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, text_emb):
        B, C, H, W = x.shape
        N_img = H * W
        N_text = text_emb.shape[1]

        x_norm = self.norm(x)
        x_flat = x_norm.view(B, C, N_img).permute(0, 2, 1)

        q = self.q_proj(x_flat)      # [B, N_img, hidden_dim]
        k = self.k_proj(text_emb)    # [B, N_text, hidden_dim]
        v = self.v_proj(text_emb)    # [B, N_text, hidden_dim]

        # Explicitly split heads and transpose to (B, num_heads, SeqLen, head_dim). This is the shape SDPA expects.
        q = q.view(B, N_img, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, N_img, head_dim]
        k = k.view(B, N_text, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, N_text, head_dim]
        v = v.view(B, N_text, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, N_text, head_dim]

        eps = 1e-6
        q = q / (q.norm(dim=-1, keepdim=True) + eps)
        k = k / (k.norm(dim=-1, keepdim=True) + eps)
        scale_factor = 1.0

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale_factor
        ) # [B, num_heads, N_img, head_dim]

        # Combine heads: transpose back and reshape to (B, N_img, hidden_dim)
        out = attn_out.transpose(1, 2).contiguous().view(B, N_img, self.hidden_dim)

        out = self.out_proj(out)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        return x + out

        
class Upsample(nn.Module):
    def __init__(self, channels, up_type="nearest"):
        """
        up_type: "nearest" (default), "transp_conv"
        """
        super().__init__()
        self.up_type = up_type

        if self.up_type == "transp_conv":
            self.up = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1) # check if channels should be doubled
        else:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        if self.up_type == "transp_conv":
            x = self.up(x)
        else:
            x = F.interpolate(x, scale_factor=2)
            x = self.conv(x)
        return x


class ResBlockD(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm=True, down_type="stride_conv"):
        """
        use_spectral_norm: True (default)
        down_type: "stride_conv" (default), "avgpool"
        """
        super().__init__()

        def conv_layer(in_c, out_c, k=3, p=1, use_sn=use_spectral_norm):
            conv = nn.Conv2d(in_c, out_c, kernel_size=k, padding=p)
            if use_sn:
                return torch.nn.utils.spectral_norm(conv)
            return conv

        self.in_layers = nn.Sequential(
            conv_layer(in_channels, out_channels, use_sn=use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.out_layers = nn.Sequential(
            conv_layer(out_channels, out_channels, use_sn=use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
            #nn.Dropout2d(0.3)
        )

        if in_channels == out_channels:
            self.skip_proj = nn.Identity()
        else:
            self.skip_proj = conv_layer(in_channels, out_channels, k=1, p=0, use_sn=use_spectral_norm)

        self.down = Downsample(out_channels, down_type)

    def forward(self, x):
        skip = self.skip_proj(x)

        out = self.in_layers(x)
        out = self.out_layers(out)

        # downsample
        skip = self.down(skip)  # downsample skip path
        out = self.down(out)    # downsample residual path

        return skip + out

class Downsample(nn.Module):
    def __init__(self, channels, down_type="stride_conv"):
        """
        down_type: "stride_conv" (default), "avgpool"
        """
        super().__init__()
        if down_type == "stride_conv":
            self.down = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        elif down_type == "avgpool":
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unsupported down_type: {down_type}")

    def forward(self, x):
        return self.down(x)
