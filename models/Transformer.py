import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpectralAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpatialSpectralBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpectralAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv3d(1, embed_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        # Input: [B, 1, C, H, W]
        x = self.proj(x)  # [B, embed_dim, C, H, W]
        return x

class HSI_SR_Transformer(nn.Module):
    def __init__(self, scale_factor=4, in_channels=1, embed_dim=16, num_heads=2, depth=1, mlp_ratio=2.):
        super().__init__()
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(in_channels, embed_dim)
        # 只用一个Transformer block，参数大幅减小
        self.block = SpatialSpectralBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop=0.,
            attn_drop=0.,
            drop_path=0.0,
            norm_layer=nn.LayerNorm)
        # 简化上采样，直接用3D插值
        self.reconstruct = nn.Conv3d(embed_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Input: [B, 1, C, H, W]
        x = self.patch_embed(x)  # [B, embed_dim, C, H, W]
        B, D, C, H, W = x.shape
        # flatten为序列
        x_seq = x.permute(0,2,3,4,1).reshape(B, C*H*W, D)
        x_seq = self.block(x_seq)
        # 恢复回3D
        x = x_seq.view(B, C, H, W, D).permute(0,4,1,2,3)  # [B, embed_dim, C, H, W]
        # 3D上采样
        x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor), mode='trilinear', align_corners=False)
        x = self.reconstruct(x)  # [B, 1, C, H', W']
        return x

# 中心裁剪函数 - 用于测试时处理
def center_crop(tensor, target_size):
    _, _, _, H, W = tensor.shape
    th, tw = target_size
    start_h = (H - th) // 2
    start_w = (W - tw) // 2
    return tensor[..., start_h:start_h+th, start_w:start_w+tw]

# 测试函数示例
def test_model():
    # 模拟输入数据 [batch, 1, channels, height, width]
    x = torch.randn(4, 1, 31, 32, 32)  # 假设是CAVE数据集的patch
    
    model = HSI_SR_Transformer(scale_factor=2)
    pred = model(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", pred.shape)  # 应该为 [4, 1, 31, 64, 64]
    
    # 测试中心裁剪
    cropped = center_crop(pred, (60, 60))
    print("Cropped shape:", cropped.shape)

if __name__ == "__main__":
    test_model()