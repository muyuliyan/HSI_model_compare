"""
MADNet3D: A practical PyTorch implementation of a 3D Multi-scale Attention
Denoising Network for Hyperspectral Image (HSI) denoising.

Why this file:
- The user asked for a detailed model-structure-only implementation suitable for
  plugging into an existing training/evaluation pipeline.
- This is an original implementation inspired by the typical design choices in
  HSI denoising literature (multi-scale 3D convolutions + attention + residual
  learning). It is not copied from any specific paper or repository.

Tensor conventions:
- Input HSI: either [B, S, H, W] or [B, 1, S, H, W]
  * S = spectral bands (depth for 3D convs)
  * We internally convert [B, S, H, W] -> [B, 1, S, H, W]
- Output: same shape as the input (we predict the clean image by residual
  learning: output = input - predicted_noise if residual=True; otherwise direct).

Key components:
- MultiScale3D: parallel 3D convs with kernel sizes (3, 5, 7) to enlarge
  the spectral-spatial receptive field; concatenation + 1x1x1 fuse.
- SE3D (channel attention): squeeze-and-excitation over (S, H, W)
  to recalibrate feature channels.
- SpectralAttention3D: depth-wise 3D conv along spectral dimension (kernel
  size (3,1,1)) to emphasize informative spectral relations.
- ResidualMSABlock3D: MultiScale3D -> (SE3D + SpectralAttention3D) -> residual.
- MADNet3D: head 3D conv -> N residual blocks -> tail 3D conv; optional
  shallow skip and global residual.

Hyperparameters you may tune:
- base_channels: 48/64/96 (balance speed vs. quality)
- num_blocks: 6/8/12 (deeper -> better but slower)
- growth_channels in SE reduction ratio
- use_norm: whether to use GroupNorm for stability on small batches
- residual: whether to learn noise residual (recommended True for denoising)

Licensed under MIT for ease of reuse in academic experiments.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------
# Utility / Normalization
# ---------------------
class GNAct3d(nn.Module):
    """GroupNorm + GELU for 3D tensors. Optional, controlled by use_norm."""
    def __init__(self, num_channels: int, num_groups: int = 8, use_norm: bool = True):
        super().__init__()
        self.use_norm = use_norm
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels) if use_norm else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(x))


# ---------------------
# Attention modules
# ---------------------
class SE3D(nn.Module):
    """Squeeze-and-Excitation for 3D feature maps.
    Aggregates global context over (S, H, W), outputs channel-wise gates.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg(x))
        return x * w


class SpectralAttention3D(nn.Module):
    """Emphasize informative spectral (depth) relations with a depth-wise 3D conv
    of kernel size (k,1,1) applied per channel, plus a residual gate.
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = (kernel_size // 2, 0, 0)
        # depth-wise over channels
        self.dw = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1),
                            padding=pad, groups=channels, bias=True)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.dw(x)
        a = self.proj(a)
        g = self.act(a)
        return x * g + x  # gated residual


# ---------------------
# Multi-scale 3D convolutional block
# ---------------------
class MultiScale3D(nn.Module):
    """Parallel 3D convolutions with different kernel sizes, then fuse.
    """
    def __init__(self, channels: int, kernels=(3, 5, 7), use_norm: bool = True):
        super().__init__()
        self.branches = nn.ModuleList()
        branch_out = []
        for k in kernels:
            pad = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size=k, padding=pad, bias=False),
                    GNAct3d(channels, use_norm=use_norm),
                )
            )
            branch_out.append(channels)
        self.fuse = nn.Sequential(
            nn.Conv3d(sum(branch_out), channels, kernel_size=1, bias=False),
            GNAct3d(channels, use_norm=use_norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.fuse(x)


class ResidualMSABlock3D(nn.Module):
    """Residual Multi-Scale + (Channel & Spectral) Attention block.
    Structure: MS-3D -> SE3D -> SpectralAttention3D -> 1x1x1 -> residual.
    """
    def __init__(self, channels: int, use_norm: bool = True, se_reduction: int = 8):
        super().__init__()
        self.ms = MultiScale3D(channels, kernels=(3, 5, 7), use_norm=use_norm)
        self.se = SE3D(channels, reduction=se_reduction)
        self.sa = SpectralAttention3D(channels, kernel_size=3)
        self.out = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            GNAct3d(channels, use_norm=use_norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.ms(x)
        x = self.se(x)
        x = self.sa(x)
        x = self.out(x)
        return x + res


# ---------------------
# The main network
# ---------------------
class MADNet3D(nn.Module):
    """3D Multi-scale Attention Denoising Network for HSI.

    Args:
        in_channels: feature channels at the input stage (default 1 for 3D convs).
        base_channels: width of the network trunk.
        num_blocks: number of ResidualMSABlock3D in the trunk.
        use_norm: enable GroupNorm in blocks.
        se_reduction: reduction ratio for SE3D.
        residual: if True, learn the noise residual (output = inp - noise_pred).
        shallow_skip: add a shallow skip from head to tail (improves stability).

    Input:
        x: [B, S, H, W] or [B, 1, S, H, W]
    Output:
        y: same shape as input
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_blocks: int = 8,
        use_norm: bool = True,
        se_reduction: int = 8,
        residual: bool = True,
        shallow_skip: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.expect_b1 = in_channels == 1

        self.head = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            GNAct3d(base_channels, use_norm=use_norm),
        )

        self.blocks = nn.Sequential(*[
            ResidualMSABlock3D(base_channels, use_norm=use_norm, se_reduction=se_reduction)
            for _ in range(num_blocks)
        ])

        self.tail = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            GNAct3d(base_channels, use_norm=use_norm),
            nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1, bias=True),
        )

        self.shallow_skip = shallow_skip
        if shallow_skip:
            self.skip = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def _ensure_5d(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Convert [B, S, H, W] -> [B, 1, S, H, W] if needed."""
        if x.dim() == 4:
            x = x.unsqueeze(1)
            return x, True
        elif x.dim() == 5:
            return x, False
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {x.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5, squeezed = self._ensure_5d(x)
        inp = x5

        h = self.head(x5)
        t = self.blocks(h)
        t = self.tail(t)

        if self.shallow_skip:
            t = t + self.skip(inp)

        if self.residual:
            out = inp - t  # predict noise, subtract
        else:
            out = t        # direct predict clean image

        # return to original shape
        if squeezed:
            out = out.squeeze(1)
        return out


# ---------------------
# Factory / convenience
# ---------------------
def madnet3d_small(**kwargs) -> MADNet3D:
    return MADNet3D(base_channels=48, num_blocks=6, **kwargs)


def madnet3d_base(**kwargs) -> MADNet3D:
    return MADNet3D(base_channels=64, num_blocks=8, **kwargs)


def madnet3d_large(**kwargs) -> MADNet3D:
    return MADNet3D(base_channels=96, num_blocks=12, **kwargs)


# ---------------------
# Quick self-test
# ---------------------
if __name__ == "__main__":
    B, S, H, W = 2, 31, 64, 64
    x = torch.randn(B, S, H, W)
    net = madnet3d_base()
    y = net(x)
    print("Input:", x.shape, "Output:", y.shape)
    params = sum(p.numel() for p in net.parameters())
    print(f"Params: {params/1e6:.2f}M")
