import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# 空间残差块（2D卷积）
# ===============================
class SpatialResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

# ===============================
# 光谱残差块（3D卷积）
# ===============================
class SpectralResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 1, kernel_size=(3,3,3), padding=1)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=(3,3,3), padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

# ===============================
# HSI-SRN 模型
# ===============================
class HSI_SRN(nn.Module):
    def __init__(self, in_channels, num_spatial_blocks=5, num_spectral_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        
        # 输入卷积，将波段映射到高维特征
        self.input_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # 空间残差块堆叠
        self.spatial_blocks = nn.Sequential(
            *[SpatialResidualBlock(64) for _ in range(num_spatial_blocks)]
        )
        
        # 光谱残差块（3D卷积）
        self.spectral_blocks = nn.Sequential(
            *[SpectralResidualBlock(1) for _ in range(num_spectral_blocks)]
        )
        
        # 输出卷积，将特征映射回原始波段数
        self.output_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: [B, C, H, W]
        out = self.input_conv(x)
        out = self.spatial_blocks(out)
        
        # 光谱处理: 先加一个 channel 维度做 3D 卷积
        out_spectral = out.unsqueeze(1)  # [B, 1, C, H, W]
        out_spectral = self.spectral_blocks(out_spectral)
        out_spectral = out_spectral.squeeze(1)  # [B, C, H, W]
        
        out = self.output_conv(out_spectral)
        return out

# ===============================
# 测试模型
# ===============================
if __name__ == "__main__":
    # 假设HSI有31个波段，大小64x64
    dummy_input = torch.randn(2, 31, 64, 64)  # [batch, channels, H, W]
    model = HSI_SRN(in_channels=31)
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
