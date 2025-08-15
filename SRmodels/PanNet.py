import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 基本 ResNet 块
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)

# -----------------------------
# PanNet 主体结构
# -----------------------------
class PanNet(nn.Module):
    def __init__(self, in_ms_channels, num_blocks=8, feat_channels=64):
        """
        in_ms_channels: 多光谱图像通道数 (例如 HSI 波段数)
        num_blocks: ResNet 块数量
        feat_channels: 中间特征通道宽度
        """
        super().__init__()
        # 初始特征映射
        self.conv_in = nn.Conv2d(in_ms_channels + 1, feat_channels, kernel_size=3, padding=1)
        # 若希望更复杂，可加入 BN / ReLU
        self.relu = nn.ReLU(inplace=True)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResBlock(feat_channels) for _ in range(num_blocks)])
        # 输出高频残差特征
        self.conv_out = nn.Conv2d(feat_channels, in_ms_channels, kernel_size=3, padding=1)

    def forward(self, up_ms, pan_hp):
        """
        up_ms: 上采样后的低分辨率多光谱图 (batch, C_ms, H, W)
        pan_hp: PAN 图像的高通成分 (batch, 1, H, W)
        """
        x = torch.cat([up_ms, pan_hp], dim=1)  # 合并通道
        feat = self.relu(self.conv_in(x))
        feat = self.res_blocks(feat)
        res = self.conv_out(feat)  # 预测多光谱高频内容
        out = up_ms + res  # 保留光谱信息 + 加入空间细节
        return out

# -----------------------------
# 中心裁剪函数
# -----------------------------
def center_crop(tensor, target_size):
    """
    对4D或5D张量进行中心裁剪。
    tensor: [B, C, H, W] 或 [B, C, D, H, W]
    target_size: (th, tw)
    """
    if tensor.dim() == 5:
        _, _, _, H, W = tensor.shape
        th, tw = target_size
        start_h = (H - th) // 2
        start_w = (W - tw) // 2
        return tensor[..., start_h:start_h+th, start_w:start_w+tw]
    elif tensor.dim() == 4:
        _, _, H, W = tensor.shape
        th, tw = target_size
        start_h = (H - th) // 2
        start_w = (W - tw) // 2
        return tensor[..., start_h:start_h+th, start_w:start_w+tw]
    else:
        raise ValueError("Unsupported tensor shape for center_crop")
