import torch
import torch.nn as nn
import torch.nn.functional as F

# 空间卷积块
class SpatialBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.conv(x) + x)

# 光谱卷积块（1D卷积沿波段方向）
class SpectralBlock(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.conv = nn.Conv1d(bands, bands, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: [B, C, H, W] -> [B, C, H*W]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W)
        out = self.relu(self.conv(x_flat))
        out = out.view(B, C, H, W)
        return out + x

# HSI-SDeCNN
class HSI_SDeCNN(nn.Module):
    def __init__(self, bands, spatial_blocks=4, spectral_blocks=2):
        super().__init__()
        self.input_conv = nn.Conv2d(bands, 64, kernel_size=3, padding=1)
        self.spatial_layers = nn.Sequential(*[SpatialBlock(64) for _ in range(spatial_blocks)])
        self.spectral_layers = nn.Sequential(*[SpectralBlock(64) for _ in range(spectral_blocks)])
        self.output_conv = nn.Conv2d(64, bands, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = self.input_conv(x)
        out = self.spatial_layers(out)
        out = self.spectral_layers(out)
        out = self.output_conv(out)
        return out

# 测试
if __name__ == "__main__":
    x = torch.randn(2, 31, 64, 64)  # [B, C, H, W]
    model = HSI_SDeCNN(bands=31)
    y = model(x)
    print(y.shape)
