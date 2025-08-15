import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """残差密集块，专为高光谱设计"""
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels + 2*growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(in_channels + 3*growth_rate, in_channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], 1))
        return x4 * 0.2 + x  # 残差缩放

class RDN_HSI(nn.Module):
    """高光谱残差密集网络"""
    def __init__(self, in_channels=1, num_blocks=16, growth_rate=32, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 浅层特征提取
        self.sfe = nn.Conv3d(in_channels, growth_rate*2, kernel_size=3, padding=1)
        
        # 残差密集块
        self.rdb_blocks = nn.ModuleList([
            ResidualDenseBlock(growth_rate*2, growth_rate)
            for _ in range(num_blocks)
        ])
        
        # 全局特征融合
        self.gff = nn.Sequential(
            nn.Conv3d(growth_rate*2 * num_blocks, growth_rate*2, kernel_size=1),
            nn.Conv3d(growth_rate*2, growth_rate*2, kernel_size=3, padding=1)
        )
        
        # 上采样重建
        self.upsample = nn.Sequential(
            nn.Conv3d(growth_rate*2, growth_rate*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(growth_rate*2, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        sfe = self.sfe(x)
        local_features = []
        x = sfe
        for block in self.rdb_blocks:
            x = block(x)
            local_features.append(x)
        
        # 全局特征融合
        global_feat = self.gff(torch.cat(local_features, 1))
        global_feat = global_feat + sfe  # 全局残差
        
        # 上采样
        out = self.upsample(global_feat)
        out = F.interpolate(out, scale_factor=(1, self.scale_factor, self.scale_factor), mode='trilinear', align_corners=False)
        return out

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

# 测试函数
def test_model():
    # 假设输入 [B, 1, C, H, W]
    x = torch.randn(2, 1, 31, 32, 32)
    model = RDN_HSI(in_channels=1, scale_factor=2)
    pred = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", pred.shape)

if __name__ == "__main__":
    test_model()