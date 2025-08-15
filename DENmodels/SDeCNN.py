# HSI_SDeCNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock2D(nn.Module):
    """2D残差卷积块"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)

class DeepEnhancementModule(nn.Module):
    """深度增强模块（多层残差块堆叠）"""
    def __init__(self, ch, num_blocks=5):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock2D(ch) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)

class HSI_SDeCNN(nn.Module):
    """
    输入: x [B, C, H, W]  (C=波段数)
    输出: 去噪后 [B, C, H, W]
    """
    def __init__(self, num_bands, base_ch=64, num_blocks=5):
        super().__init__()
        # 3D卷积提取空间–光谱特征
        self.conv3d_1 = nn.Conv3d(1, base_ch, kernel_size=(7,3,3), padding=(3,1,1), bias=False)
        self.bn3d_1 = nn.BatchNorm3d(base_ch)
        self.relu = nn.ReLU(inplace=True)

        # 压缩光谱维到2D
        self.conv3d_to_2d = nn.Conv3d(base_ch, base_ch, kernel_size=(num_bands,1,1), bias=False)

        # 深度增强模块
        self.dem = DeepEnhancementModule(base_ch, num_blocks=num_blocks)

        # 输出层
        self.conv_out = nn.Conv2d(base_ch, num_bands, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # [B, C, H, W] -> [B, 1, C, H, W]
        x_in = x.unsqueeze(1)

        # 3D卷积
        feat_3d = self.relu(self.bn3d_1(self.conv3d_1(x_in)))

        # 压缩光谱维 -> [B, base_ch, 1, H, W]
        feat_3d = self.conv3d_to_2d(feat_3d).squeeze(2)  # -> [B, base_ch, H, W]

        # 深度增强模块
        feat_2d = self.dem(feat_3d)

        # 输出残差（噪声）
        res = self.conv_out(feat_2d)  # [B, num_bands, H, W]

        # 残差学习
        out = x - res
        return out

# -------------------------
# 测试代码
# -------------------------
if __name__ == "__main__":
    B, C, H, W = 2, 31, 64, 64
    model = HSI_SDeCNN(num_bands=C, base_ch=64, num_blocks=5)
    y = model(torch.randn(B, C, H, W))
    print("Output shape:", y.shape)  # [B, C, H, W]
