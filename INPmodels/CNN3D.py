import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# 3D卷积残差块
# ===============================
class Residual3DBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 输入和输出都是 [B, 1, C, H, W]
        self.conv1 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

# ===============================
# HSI-3D-CNN 模型
# ===============================
class HSI_3D_CNN(nn.Module):
    def __init__(self, num_blocks=5):
        super().__init__()
        self.num_blocks = num_blocks
        # 输入卷积: 将原始波段作为 channel 维度加入 3D 卷积
        self.input_conv = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        # 3D残差块堆叠
        self.res_blocks = nn.Sequential(
            *[Residual3DBlock(1) for _ in range(num_blocks)]
        )
        # 输出卷积
        self.output_conv = nn.Conv3d(1, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: [B, C, H, W] -> 3D卷积需要 [B, 1, C, H, W]
        out = x.unsqueeze(1)
        out = self.input_conv(out)
        out = self.res_blocks(out)
        out = self.output_conv(out)
        # squeeze回原来的形状
        out = out.squeeze(1)  # [B, C, H, W]
        return out

# ===============================
# 测试模型
# ===============================
if __name__ == "__main__":
    dummy_input = torch.randn(2, 31, 64, 64)  # [B, C, H, W]
    model = HSI_3D_CNN(num_blocks=5)
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
