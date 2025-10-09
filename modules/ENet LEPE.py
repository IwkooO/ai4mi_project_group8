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

### EXTENSION
class OverlapPatchEmbed(nn.Module):
        def __init__(self,c_in,embed_dim,patch=4,stride=4,pad=0):
                super().__init__()
                self.proj = nn.Conv2d(c_in,embed_dim,kernel_size=patch,stride=stride,padding=pad)
                self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x: Tensor):
                x = self.proj(x) # B C H W -> B C H' W'
                _, _, H, W = x.shape
                x = x.flatten(2).transpose(1, 2) # B C H W -> B C HW -> B HW C
                x = self.norm(x)
                return x, (H, W)

class MixFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.dw  = nn.Conv2d(hid, hid, 3, padding=1, groups=hid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dw(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        return self.fc2(x)

class MHSASR(nn.Module):
        """ Multi-Head Self-Attention with Spatial Reduction """
        def __init__(self,dim,heads=4,sr_ratio=2,dropout=0.0,lepe=True):
                super().__init__()
                self.h = heads
                self.q = nn.Linear(dim,dim)
                self.kv = nn.Linear(dim,dim*2)
                self.proj= nn.Linear(dim,dim)
                self.dropout = nn.Dropout(dropout)
                self.sr_ratio = sr_ratio
                self.lepe = lepe # local positional encoding
                if sr_ratio > 1:
                        # depthwise reduce tokens spatially before making K,V
                        self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
                        self.norm = nn.LayerNorm(dim)
                if self.lepe:
                        self.lepe_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        def forward(self, x: Tensor, H: int, W: int) -> Tensor:
                B, N, C = x.shape
                d = C // self.h # dimension per head
                q = self.q(x).reshape(B, N, self.h, d).permute(0, 2, 1, 3) # B N C -> B N h (C/h) -> B h N (C/h)
        
                if self.sr_ratio > 1:   
                        xs = x.transpose(1,2).reshape(B,C,H,W) 
                        xs = self.sr(xs).flatten(2).transpose(1,2)
                        xs = self.norm(xs)
                else:
                        xs = x
                kv = self.kv(xs).reshape(B, -1, 2, self.h, d).permute(2, 0, 3, 1, 4) 
                k, v = kv[0], kv[1] # each: B h N (C/h)

                attn = (q @ k.transpose(-2, -1)) * (d ** -0.5)
                attn = attn.softmax(dim=-1)
                attn = self.dropout(attn)

                out = (attn @ v).transpose(1, 2).reshape(B, N, C)

                if self.lepe:
                        x_hw = x.transpose(1, 2).reshape(B, C, H, W)    # B,C,H,W
                        pe = self.lepe_dw(x_hw).flatten(2).transpose(1, 2)  # B,N,C
                        out = out + pe
                out = self.proj(out)
                return out

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class ViTBlock(nn.Module):
        def __init__(self, dim, heads=4, sr_ratio=2, mlp_ratio=4, dropout=0.0, drop_path=0.1):
                super().__init__()
                self.n1 = nn.LayerNorm(dim)
                self.attn = MHSASR(dim, heads=heads, sr_ratio=sr_ratio, dropout=dropout)
                self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                self.n2 = nn.LayerNorm(dim)
                self.ffn = MixFFN(dim, mlp_ratio=mlp_ratio)

        def forward(self, x, H, W):
                x = x + self.drop_path(self.attn(self.n1(x), H, W))
                x = x + self.drop_path(self.ffn(self.n2(x), H, W))
                return x

class TransformerBottleneck(nn.Module):
        """Goes into the deepest part of the ENet.

        Added support for progressive stochastic depth via `drop_path_max`:
        if drop_path_max>0 and depth>1, the residual branches in successive ViTBlocks
        use linearly increasing drop probabilities from 0 -> drop_path_max.
        """
        def __init__(self, c_in, embed_dim=256, depth=2, heads=4, sr_ratio=2, patch=4, drop_path_max: float | None = None):
                super().__init__()

                self.patch = OverlapPatchEmbed(c_in, embed_dim, patch=patch, stride=patch)

                # Determine per-block drop_path values
                if drop_path_max is None:
                        # Preserve previous behaviour (uniform 0.1 in each block)
                        dpr = [0.1] * depth
                else:
                        if drop_path_max > 0 and depth > 1:
                                dpr = torch.linspace(0, drop_path_max, steps=depth).tolist()
                        else:
                                dpr = [0.0] * depth

                self.blocks = nn.ModuleList([
                        ViTBlock(embed_dim, heads=heads, sr_ratio=sr_ratio, drop_path=dpr[i]) for i in range(depth)
                ])
                self.proj_back = nn.Conv2d(embed_dim, c_in, kernel_size=1, bias=False)
                self.bn = nn.BatchNorm2d(c_in)

                # gated fusion to keep CNN path dominant when needed
                self.gate = nn.Sequential(
                        nn.Conv2d(2 * c_in, c_in, kernel_size=1, bias=False),
                        nn.BatchNorm2d(c_in),
                        nn.Sigmoid()
                )

        def forward(self, x):  # x: [B,C,H,W]
                B, C, H, W = x.shape
                tok, (h, w) = self.patch(x)                          # [B, h*w, E]
                for blk in self.blocks:
                        tok = blk(tok, h, w)
                feat = tok.transpose(1, 2).reshape(B, -1, h, w)      # [B,E,h,w]
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
                feat = self.bn(self.proj_back(feat))                 # [B,C,H,W]
                g = self.gate(torch.cat([x, feat], dim=1))           # [B,C,H,W]
                print(f"Gate min: {g.min().item():.4f}, max: {g.max().item():.4f}, mean: {g.mean().item():.4f}")

                return x + g * feat


### END EXTENSION

def random_weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


def conv_block(in_dim, out_dim, **kwconv):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, **kwconv),
                             nn.BatchNorm2d(out_dim),
                             nn.PReLU())


def conv_block_asym(in_dim, out_dim, *, kernel_size: int):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim,
                                       kernel_size=(kernel_size, 1),
                                       padding=(2, 0)),
                             nn.Conv2d(out_dim, out_dim,
                                       kernel_size=(1, kernel_size),
                                       padding=(0, 2)),
                             nn.BatchNorm2d(out_dim),
                             nn.PReLU())


class BottleNeck(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor,
                     *, dropoutRate=0.01, dilation=1,
                     asym: bool = False, dilate_last: bool = False):
                super().__init__()
                self.in_dim = in_dim
                self.out_dim = out_dim
                mid_dim: int = in_dim // projectionFactor

                # Main branch

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=1)

                if not asym:
                        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=dilation, dilation=dilation)
                else:
                        self.block1 = conv_block_asym(mid_dim, mid_dim, kernel_size=5)

                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                self.do = nn.Dropout(p=dropoutRate)
                self.PReLU_out = nn.PReLU()

                if in_dim > out_dim:
                        self.conv_out = conv_block(in_dim, out_dim, kernel_size=1)
                elif dilate_last:
                        self.conv_out = conv_block(in_dim, out_dim, kernel_size=3, padding=1)
                else:
                        self.conv_out = nn.Identity()

        def forward(self, in_) -> Tensor:
                # Main branch
                # Secondary branch
                b0 = self.block0(in_)
                b1 = self.block1(b0)
                b2 = self.block2(b1)
                do = self.do(b2)

                output = self.PReLU_out(self.conv_out(in_) + do)

                return output


class BottleNeckDownSampling(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor):
                super().__init__()
                mid_dim: int = in_dim // projectionFactor

                # Main branch
                self.maxpool0 = nn.MaxPool2d(2, return_indices=True)

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=2, padding=0, stride=2)
                self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                # Regularizer
                self.do = nn.Dropout(p=0.01)
                self.PReLU = nn.PReLU()

                # Out

        def forward(self, in_) -> tuple[Tensor, Tensor]:
                # Main branch
                maxpool_output, indices = self.maxpool0(in_)

                # Secondary branch
                b0 = self.block0(in_)
                b1 = self.block1(b0)
                b2 = self.block2(b1)
                do = self.do(b2)

                _, c, _, _ = maxpool_output.shape
                output = do
                output[:, :c, :, :] += maxpool_output

                final_output = self.PReLU(output)

                return final_output, indices


class BottleNeckUpSampling(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor):
                super().__init__()
                mid_dim: int = in_dim // projectionFactor

                # Main branch
                self.unpool = nn.MaxUnpool2d(2)

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=3, padding=1)
                self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                # Regularizer
                self.do = nn.Dropout(p=0.01)
                self.PReLU = nn.PReLU()

                # Out

        def forward(self, args) -> Tensor:
                # nn.Sequential cannot handle multiple parameters:
                in_, indices, skip = args

                # Main branch
                up = self.unpool(in_, indices)

                # Secondary branch
                b0 = self.block0(torch.cat((up, skip), dim=1))
                b1 = self.block1(b0)
                b2 = self.block2(b1)
                do = self.do(b2)

                output = self.PReLU(up + do)

                return output


class ENet(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, **kwargs):
                super().__init__()
                F: int = kwargs["factor"] if "factor" in kwargs else 4  # Projecting factor
                K: int = kwargs["kernels"] if "kernels" in kwargs else 16  # n_kernels

                # from models.enet import (BottleNeck,
                #                          BottleNeckDownSampling,
                #                          BottleNeckUpSampling,
                #                          conv_block)

                # Initial operations
                self.conv0 = nn.Conv2d(in_dim, K - 1, kernel_size=3, stride=2, padding=1)
                self.maxpool0 = nn.MaxPool2d(2, return_indices=False, ceil_mode=False)

                # Downsampling half
                self.bottleneck1_0 = BottleNeckDownSampling(K, K * 4, F)
                self.bottleneck1_1 = nn.Sequential(BottleNeck(K * 4, K * 4, F),
                                                   BottleNeck(K * 4, K * 4, F),
                                                   BottleNeck(K * 4, K * 4, F),
                                                   BottleNeck(K * 4, K * 4, F))
                self.bottleneck2_0 = BottleNeckDownSampling(K * 4, K * 8, F)
                self.bottleneck2_1 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
                                                   BottleNeck(K * 8, K * 8, F, dilation=2),
                                                   BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
                                                   BottleNeck(K * 8, K * 8, F, dilation=4),
                                                   BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
                                                   BottleNeck(K * 8, K * 8, F, dilation=8),
                                                   BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
                                                   BottleNeck(K * 8, K * 8, F, dilation=16))
                ### EXTENSION
                # Adding transformer block also before compression to gather global cues
                # This is more lightweight than after compression version
                # self.trans_enc2 = TransformerBottleneck(
                #         c_in=K*4, embed_dim=128, depth=1, heads=2, sr_ratio=4, patch=4)

                # Main ViT block in the bottleneck
                #self.trans_mid = TransformerBottleneck(c_in=K * 8, embed_dim=256, depth=2, heads=4, sr_ratio=2, patch=4)
                self.trans_mid = TransformerBottleneck(c_in=K * 8, embed_dim=192, depth=3, heads=6, sr_ratio=1, patch=2, drop_path_max=0.15)
                ### END EXTENSION

                # Middle operations
                self.bottleneck3 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
                                                 BottleNeck(K * 8, K * 8, F, dilation=2),
                                                 BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
                                                 BottleNeck(K * 8, K * 8, F, dilation=4),
                                                 BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
                                                 BottleNeck(K * 8, K * 8, F, dilation=8),
                                                 BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
                                                 BottleNeck(K * 8, K * 4, F, dilation=16, dilate_last=True))

                # Upsampling half
                self.bottleneck4 = nn.Sequential(BottleNeckUpSampling(K * 8, K * 4, F),
                                                 BottleNeck(K * 4, K * 4, F, dropoutRate=0.1),
                                                 BottleNeck(K * 4, K, F, dropoutRate=0.1))
                self.bottleneck5 = nn.Sequential(BottleNeckUpSampling(K * 2, K, F),
                                                 BottleNeck(K, K, F, dropoutRate=0.1))

                # Final upsampling and covolutions
                self.final = nn.Sequential(conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
                                           conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
                                           nn.Conv2d(K, out_dim, kernel_size=1))

                print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

        def forward(self, input):
                # Initial operations
                conv_0 = self.conv0(input)
                maxpool_0 = self.maxpool0(input)
                outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

                # Downsampling half
                bn1_0, indices_1 = self.bottleneck1_0(outputInitial)
                bn1_out = self.bottleneck1_1(bn1_0)
                ### EXTENSION 
                #bn1_out = self.trans_enc2(bn1_out)
                ### END EXTENSION
                bn2_0, indices_2 = self.bottleneck2_0(bn1_out)
                bn2_out = self.bottleneck2_1(bn2_0)
                ### EXTENSION
                x_mid = self.trans_mid(bn2_out)
                # Middle operations
                bn3_out =self.bottleneck3(x_mid)
                ### END EXTENSION

                # Middle operations
                #bn3_out = self.bottleneck3(bn2_out)

                # Upsampling half
                bn4_out = self.bottleneck4((bn3_out, indices_2, bn1_out))
                bn5_out = self.bottleneck5((bn4_out, indices_1, outputInitial))

                # Final upsampling and covolutions
                interpolated = F.interpolate(bn5_out, mode='nearest', scale_factor=2)
                return self.final(interpolated)

        def init_weights(self, *args, **kwargs):
                self.apply(random_weights_init)
