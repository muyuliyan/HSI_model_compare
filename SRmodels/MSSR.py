import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualAttention(nn.Module):
    """增强的残差注意力模块，适用于超分辨率任务"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        channel_att = self.channel_attention(x)
        return x * channel_att + x

class MultiScaleConv(nn.Module):
    """改进的多尺度卷积模块，保留空间信息"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用不同深度的卷积核捕获多尺度特征
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1))
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding=1)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=(5,3,3), padding=(2,1,1))
        
        # 合并后的处理
        self.bn = nn.BatchNorm3d(out_channels*3)
        self.relu = nn.ReLU(inplace=True)
        
        # 减少通道数的瓶颈层
        self.bottleneck = nn.Conv3d(out_channels*3, out_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.conv1(x)  # 空间特征
        out2 = self.conv2(x)  # 光谱-空间特征
        out3 = self.conv3(x)  # 更大感受野的光谱-空间特征
        
        # 合并特征并处理
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.relu(self.bn(out))
        out = self.bottleneck(out)  # 减少通道数
        
        return out

class UpsampleBlock(nn.Module):
    """高效的亚像素上采样模块"""
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 使用亚像素卷积进行高效上采样
        self.conv = nn.Conv3d(in_channels, in_channels * (scale_factor ** 2), 
                             kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        # 对空间维度进行上采样
        x = self.conv(x)
        
        # 重新排列维度以使用PixelShuffle
        # PixelShuffle要求输入为[B, C, H, W]，所以需要合并光谱和通道维度
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, D, C, H, W]
        x = x.view(B, D * C, H, W)  # [B, D*C, H, W]
        
        # 应用PixelShuffle进行空间上采样
        x = self.pixel_shuffle(x)  # [B, D*C/(scale^2), H*scale, W*scale]
        
        # 恢复原始维度结构
        x = x.view(B, D, C // (self.scale_factor ** 2), 
                  H * self.scale_factor, W * self.scale_factor)
        x = x.permute(0, 2, 1, 3, 4)  # [B, C', D, H', W']
        
        return self.activation(x)

class MSSR(nn.Module):
    """用于高光谱图像超分辨率的MSSR模型"""
    def __init__(self, in_channels=1, out_channels=1, scale_factor=2, num_features=32):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 初始特征提取
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 光谱分支 - 强调光谱信息
        self.spectral_branch = nn.Sequential(
            MultiScaleConv(num_features, num_features),
            ResidualAttention(num_features),
            MultiScaleConv(num_features, num_features),
            ResidualAttention(num_features)
        )
        
        # 空间分支 - 强调空间信息
        self.spatial_branch = nn.Sequential(
            MultiScaleConv(num_features, num_features),
            ResidualAttention(num_features),
            MultiScaleConv(num_features, num_features),
            ResidualAttention(num_features)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv3d(num_features * 2, num_features, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualAttention(num_features)
        )
        
        # 上采样模块
        self.upsample = UpsampleBlock(num_features, scale_factor=scale_factor)
        
        # 重建模块
        self.reconstruction = nn.Sequential(
            nn.Conv3d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(num_features, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # 初始特征提取
        x = self.init_conv(x)
        
        # 并行处理光谱和空间分支
        spectral_feat = self.spectral_branch(x)
        spatial_feat = self.spatial_branch(x)
        
        # 特征融合
        fused = torch.cat([spectral_feat, spatial_feat], dim=1)
        fused = self.fusion(fused)
        
        # 上采样
        upsampled = self.upsample(fused)
        
        # 重建
        out = self.reconstruction(upsampled)
        return out

# 测试函数
def test_model():
    # 模拟输入数据 [batch, channels, depth, height, width]
    x = torch.randn(4, 1, 31, 32, 32)  # 低分辨率输入
    
    model = MSSR(in_channels=1, scale_factor=2)
    pred = model(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", pred.shape)  # 应该为 [4, 1, 31, 64, 64]

def center_crop(tensor, target_size):
    """
    对4D或5D张量进行中心裁剪。
    tensor: [B, C, D, H, W] 或 [B, C, H, W]
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

if __name__ == "__main__":
    test_model()