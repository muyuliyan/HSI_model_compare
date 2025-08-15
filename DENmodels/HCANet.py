# hcanet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# -------------------------
# helpers
# -------------------------
def channel_shuffle(x, groups):
    # x: [B, C, H, W] , groups divides C
    B, C, H, W = x.shape
    assert C % groups == 0
    x = x.view(B, groups, C // groups, H, W)
    x = x.permute(0,2,1,3,4).contiguous()
    return x.view(B, C, H, W)

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_ch, k=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=stride,
                            padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.GELU()
    def forward(self,x):
        return self.act(self.bn(self.dw(x)))

# -------------------------
# CAFM (Convolution + Attention Fusion Module)
# -------------------------
class CAFM(nn.Module):
    """
    Input x: [B, C, H, W]  (we will use 2D features; in paper some 3D convs used earlier)
    Implements local branch (1x1 -> channel shuffle -> group depthwise/3x3) and
    global branch (Q,K via 1x1 & depthwise conv, attention computed as in paper).
    """
    def __init__(self, channels, groups=4):
        super().__init__()
        self.channels = channels
        self.groups = groups
        # local branch
        self.local_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.local_dw = DepthwiseConv2d(channels, k=3, padding=1)
        # 3x3x3 in paper used in some places; we implement core idea in 2D for channel mixing.
        # global branch
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                              groups=channels, bias=False)  # depthwise 3x3
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # final fusion
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # learnable scale alpha as in paper
        self.alpha = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        B,C,H,W = x.shape
        # local branch
        t = self.local_proj(x)
        t = channel_shuffle(t, self.groups)
        local = self.local_dw(t)  # [B,C,H,W]

        # global branch: form Q (HW x C), K (C x HW) to get CxC attention
        Q = self.q_proj(x)                   # [B,C,H,W]
        K = self.k_dw(x)                     # [B,C,H,W] (depthwise processed)
        V = self.v_proj(x)                   # [B,C,H,W]

        # reshape: Q_hat: [B, HW, C], K_hat: [B, C, HW], V_hat: [B, C, HW]
        Q_hat = Q.flatten(2).permute(0,2,1)  # [B, HW, C]
        K_hat = K.flatten(2)                 # [B, C, HW]
        V_hat = V.flatten(2)                 # [B, C, HW]

        # attention: A = softmax( K_hat @ Q_hat / alpha )  -> [B, C, C]
        # compute KQ: [B, C, C]
        KQ = torch.matmul(K_hat, Q_hat) / (self.alpha + 1e-6)
        A = F.softmax(KQ, dim=-1)  # along last dim (C)
        # apply to V: out_flat = V_hat @ A   ; V_hat: [B, C, HW], A: [B, C, C] -> [B, C, HW]
        out_flat = torch.matmul(A, V_hat)  # [B, C, HW]
        Fatt = out_flat.view(B, C, H, W)   # [B,C,H,W]

        # sum local + attention, then final proj
        out = local + Fatt
        out = self.out_proj(out)
        return out

# -------------------------
# MSFN (Multi-Scale Feed-Forward Network)
# -------------------------
class MSFN(nn.Module):
    """
    paper: expand channels (gamma=2), split into two paths:
      - upper: multi-scale dilated convs (dilations 2 & 3) + gating
      - lower: depthwise conv
    gate: element-wise product
    """
    def __init__(self, channels, expand_ratio=2):
        super().__init__()
        hidden = channels * expand_ratio
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        # upper path: dilated convs
        self.dil_conv1 = nn.Conv2d(hidden//2, hidden//2, kernel_size=3, padding=2, dilation=2, bias=False)
        self.dil_conv2 = nn.Conv2d(hidden//2, hidden//2, kernel_size=3, padding=3, dilation=3, bias=False)
        # lower path: depthwise
        self.dw = nn.Conv2d(hidden//2, hidden//2, kernel_size=3, padding=1, groups=hidden//2, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden//2, channels, kernel_size=1, bias=False)
        self.fc2b = nn.Conv2d(hidden//2, channels, kernel_size=1, bias=False)
    def forward(self, x):
        # x: [B,C,H,W]
        z = self.fc1(x)  # [B, hidden, H, W]
        # split channels
        c = z.shape[1]//2
        up = z[:, :c, :, :]
        low = z[:, c:, :, :]
        # upper: two dilated convs then combine
        up = self.act(self.dil_conv1(up))
        up = self.act(self.dil_conv2(up))
        # lower: depthwise
        low = self.act(self.dw(low))
        # gating: element-wise product (after projecting to same shape)
        up_proj = self.fc2(up)    # -> [B,C,H,W]
        low_proj = self.fc2b(low) # -> [B,C,H,W]
        out = up_proj * low_proj  # gating
        return out

# -------------------------
# CAMixing block (CAFM + MSFN + residual & norms)
# -------------------------
class CAMixingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # layernorm: operate on channel dimension -> use LayerNorm over (H,W,C) via permute
        self.norm1 = nn.LayerNorm(channels)
        self.cafm = CAFM(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.msfn = MSFN(channels)
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))

    def _ln(self, x, ln):
        # x: [B,C,H,W] -> permute to [B,H,W,C] for LayerNorm with normalized_shape=C
        x_perm = x.permute(0,2,3,1)
        x_norm = ln(x_perm)
        return x_norm.permute(0,3,1,2)

    def forward(self, x):
        # CAFM with residual + LN scaling
        y = self._ln(x, self.norm1)
        y = self.cafm(y)
        x = x + self.gamma1 * y
        # MSFN
        y2 = self._ln(x, self.norm2)
        y2 = self.msfn(y2)
        x = x + self.gamma2 * y2
        return x

# -------------------------
# Simple U-shaped HCANet backbone
# -------------------------
class HCANet(nn.Module):
    """
    Simplified U-shaped stack of CAMixing blocks.
    Input: x [B, C, H, W]  (C = bands)
    For HSI, paper uses some 3D conv early stage; here we use a 3D->2D conversion:
      - initial 3x3x3 conv applied on [B,1,C,H,W] simulated by unfolding or group conv.
    For practicality, we implement a lightweight variant that follows paper's block design.
    """
    def __init__(self, in_bands:int, base_ch=64, num_layers=4):
        super().__init__()
        self.in_bands = in_bands
        # initial 3D-like conv: emulate with Conv3d then squeeze
        self.conv3d = nn.Conv3d(1, base_ch, kernel_size=(3,3,3), padding=(1,1,1), bias=False)
        self.bn3d = nn.BatchNorm3d(base_ch)
        # project 3D -> 2D: conv with kernel (bands,1,1) to collapse spectral dim
        self.spec_pool = nn.Conv3d(base_ch, base_ch, kernel_size=(in_bands,1,1), bias=False)
        # encoder: downsample spatially via conv stride2
        self.encs = nn.ModuleList()
        self.decs = nn.ModuleList()
        ch = base_ch
        for i in range(num_layers):
            self.encs.append(nn.Sequential(
                CAMixingBlock(ch),
                nn.Conv2d(ch, ch*2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch*2),
                nn.GELU()
            ))
            ch *= 2
        # bottleneck
        self.bottleneck = nn.Sequential(CAMixingBlock(ch), CAMixingBlock(ch))
        # decoder
        for i in range(num_layers):
            self.decs.append(nn.Sequential(
                nn.ConvTranspose2d(ch, ch//2, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(ch//2),
                nn.GELU(),
                CAMixingBlock(ch//2)
            ))
            ch = ch//2
        # head to predict residual noise map with same channels as bands
        self.head = nn.Conv2d(base_ch, in_bands, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # x: [B, bands, H, W]
        B, BANDS, H, W = x.shape
        x3d = x.unsqueeze(1)  # [B,1,BANDS,H,W]
        feat3d = self.bn3d(self.conv3d(x3d))  # [B, base_ch, BANDS, H, W]
        # collapse spectral dim using conv with kernel size BANDS
        feat2d = self.spec_pool(feat3d).squeeze(2)  # [B, base_ch, H, W]
        skips = []
        out = feat2d
        for enc in self.encs:
            skips.append(out)  # 先保存
            out = enc(out)
        out = self.bottleneck(out)
        # decode with skip connections
        for dec in self.decs:
            skip = skips.pop()
            out = dec(out)
            # align shapes possibly
            if out.shape != skip.shape:
                # crop/pad to match
                min_h = min(out.shape[2], skip.shape[2])
                min_w = min(out.shape[3], skip.shape[3])
                out = out[:, :, :min_h, :min_w]
                skip = skip[:, :, :min_h, :min_w]
            out = out + skip
        res = self.head(out)  # [B, bands, H, W]
        # residual learning: paper used I_hat = I + IN  where IN predicted noise residual
        return x + res  # clean estimate

# -------------------------
# Quick smoke test
# -------------------------
if __name__ == "__main__":
    model = HCANet(in_bands=31, base_ch=32, num_layers=3)
    x = torch.randn(2, 31, 128, 128)
    y = model(x)
    print("in:", x.shape, "out:", y.shape)
