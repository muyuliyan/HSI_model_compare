
import torch
import torch.nn as nn
import torch.nn.functional as F

# Patch Embedding: 3D卷积实现
class PatchEmbed3D(nn.Module):
	def __init__(self, in_channels=1, embed_dim=48, patch_size=3):
		super().__init__()
		self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, padding=patch_size//2)
	def forward(self, x):
		return self.proj(x)

# 简单的3D多头自注意力
class MultiHeadSelfAttention3D(nn.Module):
	def __init__(self, dim, num_heads=4, attn_drop=0., proj_drop=0.):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5
		self.qkv = nn.Linear(dim, dim * 3, bias=True)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)
	def forward(self, x):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
		q, k, v = qkv[0], qkv[1], qkv[2]
		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x = (attn @ v).transpose(1,2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x

# Transformer Block
class TransformerBlock3D(nn.Module):
	def __init__(self, dim, num_heads=4, mlp_ratio=2., drop=0., attn_drop=0.):
		super().__init__()
		self.norm1 = nn.LayerNorm(dim)
		self.attn = MultiHeadSelfAttention3D(dim, num_heads, attn_drop, drop)
		self.norm2 = nn.LayerNorm(dim)
		self.mlp = nn.Sequential(
			nn.Linear(dim, int(dim*mlp_ratio)),
			nn.GELU(),
			nn.Linear(int(dim*mlp_ratio), dim),
			nn.Dropout(drop)
		)
	def forward(self, x):
		x = x + self.attn(self.norm1(x))
		x = x + self.mlp(self.norm2(x))
		return x

# 主体网络
class HSIDenoiseTransformer(nn.Module):
	"""
	高光谱去噪Transformer模型，结构参考主流医学/高光谱去噪论文。
	输入/输出: [B, 1, C, H, W]，支持噪声残差学习。
	"""
	def __init__(self, in_channels=1, embed_dim=48, depth=4, num_heads=4, mlp_ratio=2., residual=True):
		super().__init__()
		self.residual = residual
		self.patch_embed = PatchEmbed3D(in_channels, embed_dim)
		self.blocks = nn.Sequential(*[
			TransformerBlock3D(embed_dim, num_heads, mlp_ratio)
			for _ in range(depth)
		])
		self.norm = nn.LayerNorm(embed_dim)
		self.reconstruct = nn.Conv3d(embed_dim, in_channels, kernel_size=3, padding=1)
	def forward(self, x):
		# x: [B, 1, C, H, W]
		inp = x
		fea = self.patch_embed(x)  # [B, embed_dim, C, H, W]
		B, D, C, H, W = fea.shape
		x_seq = fea.permute(0,2,3,4,1).reshape(B, C*H*W, D)
		x_seq = self.blocks(x_seq)
		x_seq = self.norm(x_seq)
		fea = x_seq.view(B, C, H, W, D).permute(0,4,1,2,3)
		out = self.reconstruct(fea)
		if self.residual:
			out = inp - out
		return out

# 测试
if __name__ == "__main__":
	x = torch.randn(2, 1, 31, 32, 32)
	model = HSIDenoiseTransformer(in_channels=1, embed_dim=48, depth=4, num_heads=4)
	y = model(x)
	print("Input:", x.shape, "Output:", y.shape)
	params = sum(p.numel() for p in model.parameters())
	print(f"Params: {params/1e6:.2f}M")
