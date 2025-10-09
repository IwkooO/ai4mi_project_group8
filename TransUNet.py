#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Jose Dolz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Refined TransUNet (2D) 
# - CNN encoder to 1/16 resolution
# - ViT bottleneck on the 1/16 feature map (tokens + abs PE, no CLS)
# - UNet decoder with skip concatenations
# - Prints trainable parameter counts


# -----------------------------
# Small UNet-style building blocks
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample (bilinear) + concat skip + double conv"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# -----------------------------
# ViT bottleneck on 1/16 feature map (no CLS)
# -----------------------------
class ViTBottleneckOnMap(nn.Module):
    """
    Projects 1/16 CNN feature map to tokens, adds abs positional embeddings,
    runs Transformer encoder (full attention), projects back, and residual-adds.
    """
    def __init__(
        self,
        in_channels: int,            # channels at 1/16 stage (e.g., base_ch*8)
        embed_dim: int = 768,        # ViT width 
        depth: int = 12,             # number of Transformer layers
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        base_pe_hw: int = 14,        # base grid for learnable abs PE (interpolated)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.base_pe_hw = base_pe_hw

        # Project to embedding space
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.norm_in = nn.LayerNorm(embed_dim)

        # Transformer encoder (vanilla PyTorch)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Learnable absolute position embedding (initialized at base_pe_hw x base_pe_hw)
        self.pos_embed = nn.Parameter(torch.zeros(1, base_pe_hw * base_pe_hw, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Back projection to CNN channels + BN, then residual add
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(in_channels)

    @torch.no_grad()
    def _interp_pos(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """Interpolate abs PE from (B x B) to (H x W), return (1, H*W, E)."""
        B = self.base_pe_hw
        pe_2d = self.pos_embed.transpose(1, 2).reshape(1, self.embed_dim, B, B)  # (1,E,B,B)
        pe_2d = F.interpolate(pe_2d.to(device=device, dtype=dtype),
                              size=(H, W), mode="bilinear", align_corners=False)
        pe_tokens = pe_2d.reshape(1, self.embed_dim, H * W).transpose(1, 2)      # (1,N,E)
        return pe_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W) — H=W≈(input/16). Returns same shape.
        """
        B, C, H, W = x.shape
        feat = self.proj_in(x)                         # (B, E, H, W)
        tokens = feat.flatten(2).transpose(1, 2)       # (B, N, E), N=H*W
        tokens = self.norm_in(tokens)

        # absolute PE (interpolated to current grid)
        pe = self._interp_pos(H, W, x.device, tokens.dtype)  # (1, N, E)
        tokens = tokens + pe

        # Transformer
        tokens = self.transformer(tokens)              # (B, N, E)

        # back to feature map + residual
        out = tokens.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        out = self.bn_out(self.proj_out(out))
        return x + out


# -----------------------------
# TransUNet (2D)
# -----------------------------
class TransUNet2D(nn.Module):
    """
    Correct TransUNet (2D):
    - Encoder to 1/16 resolution
    - ViT bottleneck on the 1/16 map (tokens + abs PE, no CLS)
    - Decoder with CNN skip concatenations (from 1/8, 1/4, 1/2)
    """
    def __init__(
        self,
        in_ch: int = 1,           # more than 1 would be 2.5D input
        num_classes: int = 5,     
        base_ch: int = 32,        
        vit_embed_dim: int = 384, 
        vit_depth: int = 8,
        vit_heads: int = 6,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.0,
    ):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch, base_ch)                # 1/1
        self.pool = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_ch, base_ch * 2)          # 1/2
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)      # 1/4
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)      # 1/8
        self.enc5 = DoubleConv(base_ch * 8, base_ch * 8)      # 1/16 (keep channels)

        # ViT bottleneck on 1/16 map
        self.vit = ViTBottleneckOnMap(
            in_channels=base_ch * 8,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
            base_pe_hw=14,
        )

        # Decoder
        self.up4 = UpBlock(in_ch=base_ch * 8, skip_ch=base_ch * 8, out_ch=base_ch * 4)  # 1/16->1/8
        self.up3 = UpBlock(in_ch=base_ch * 4, skip_ch=base_ch * 4, out_ch=base_ch * 2)  # 1/8->1/4
        self.up2 = UpBlock(in_ch=base_ch * 2, skip_ch=base_ch * 2, out_ch=base_ch)      # 1/4->1/2
        self.up1 = UpBlock(in_ch=base_ch,     skip_ch=base_ch,     out_ch=base_ch)      # 1/2->1/1

        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

        self._init_weights()
        self._print_param_counts()   # prints params when created

    # ------------- utilities -------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _count_trainable_params(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def _print_param_counts(self):
        total = self._count_trainable_params(self)

        cnn_modules = nn.ModuleList([self.enc1, self.enc2, self.enc3, self.enc4, self.enc5,
                                     self.up4, self.up3, self.up2, self.up1, self.head])
        vit_modules = nn.ModuleList([self.vit])

        cnn_params = self._count_trainable_params(cnn_modules)
        vit_params = self._count_trainable_params(vit_modules)

        print(f"[TransUNet2D] Trainable parameters: total={total/1e6:.2f}M "
              f"(CNN={cnn_params/1e6:.2f}M, ViT={vit_params/1e6:.2f}M)")

    # ------------- forward -------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Encoder to 1/16
        e1 = self.enc1(x)       # 1/1
        p1 = self.pool(e1)      # 1/2

        e2 = self.enc2(p1)      # 1/2 features
        p2 = self.pool(e2)      # 1/4

        e3 = self.enc3(p2)      # 1/4 features
        p3 = self.pool(e3)      # 1/8

        e4 = self.enc4(p3)      # 1/8 features
        p4 = self.pool(e4)      # 1/16

        e5 = self.enc5(p4)      # 1/16, channels kept (base_ch*8)

        # ViT bottleneck
        b = self.vit(e5)        # 1/16

        # Decoder with CNN skips
        d4 = self.up4(b,  e4)   # -> 1/8
        d3 = self.up3(d4, e3)   # -> 1/4
        d2 = self.up2(d3, e2)   # -> 1/2
        d1 = self.up1(d2, e1)   # -> 1/1

        logits = self.head(d1)
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits