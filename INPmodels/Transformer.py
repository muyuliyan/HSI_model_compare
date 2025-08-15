import torch
import torch.nn as nn

class SpectralAttentionBlock(nn.Module):
    def __init__(self, bands, nhead=4, dim_feedforward=128):
        super().__init__()
        self.bands = bands
        self.flatten = nn.Flatten(2)  # [B, C, H, W] -> [B, C, H*W]
        self.transformer = nn.TransformerEncoderLayer(d_model=bands, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = self.flatten(x).transpose(1, 2)  # [B, H*W, C]
        out = self.transformer(x_flat)  # [B, H*W, C]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out + x

class HSI_Transformer(nn.Module):
    def __init__(self, bands, n_layers=2, nhead=4, dim_feedforward=128):
        super().__init__()
        self.input_conv = nn.Conv2d(bands, bands, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[SpectralAttentionBlock(bands, nhead, dim_feedforward) for _ in range(n_layers)])
        self.output_conv = nn.Conv2d(bands, bands, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = self.input_conv(x)
        out = self.blocks(out)
        out = self.output_conv(out)
        return out

# 测试
if __name__ == "__main__":
    x = torch.randn(2, 31, 64, 64)
    model = HSI_Transformer(bands=31)
    y = model(x)
    print(y.shape)
