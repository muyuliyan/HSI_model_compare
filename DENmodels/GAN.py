
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 基础积木 ----------
class SpectralSE(nn.Module):
	def __init__(self, channels, reduction=8):
		super().__init__()
		self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
		mid = max(channels // reduction, 8)
		self.fc = nn.Sequential(
			nn.Conv3d(channels, mid, kernel_size=1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv3d(mid, channels, kernel_size=1, bias=True),
			nn.Sigmoid()
		)
	def forward(self, x):
		w = self.avg_pool(x)
		w = self.fc(w)
		return x * w

class RRDB3D(nn.Module):
	def __init__(self, nf=32, growth=16, use_spectral_se=True):
		super().__init__()
		self.conv1 = nn.Conv3d(nf, growth, 3, 1, 1)
		self.conv2 = nn.Conv3d(nf + growth, growth, 3, 1, 1)
		self.conv3 = nn.Conv3d(nf + 2 * growth, nf, 3, 1, 1)
		self.act = nn.LeakyReLU(0.2, inplace=True)
		self.se = SpectralSE(nf) if use_spectral_se else nn.Identity()
	def forward(self, x):
		c1 = self.act(self.conv1(x))
		c2_in = torch.cat([x, c1], dim=1)
		c2 = self.act(self.conv2(c2_in))
		c3_in = torch.cat([x, c1, c2], dim=1)
		c3 = self.conv3(c3_in)
		out = x + 0.2 * self.se(c3)
		return out

# ---------- 生成器（去噪版） ----------
class HSIDenoiseGenerator(nn.Module):
	"""
	3D GAN 生成器（去噪版）：3D卷积主干+RRDB3D堆叠，输出与输入同尺寸
	输入/输出: [B, 1, C_bands, H, W]  -> [B, 1, C_bands, H, W]
	"""
	def __init__(self, in_ch=1, base_nf=32, num_blocks=4, growth=16, use_spectral_se=True):
		super().__init__()
		self.head = nn.Conv3d(in_ch, base_nf, 3, 1, 1)
		blocks = [RRDB3D(nf=base_nf, growth=growth, use_spectral_se=use_spectral_se)
				  for _ in range(num_blocks)]
		self.body = nn.Sequential(*blocks)
		self.body_tail = nn.Conv3d(base_nf, base_nf, 3, 1, 1)
		self.tail = nn.Sequential(
			nn.Conv3d(base_nf, base_nf, 3, 1, 1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv3d(base_nf, 1, 3, 1, 1)
		)
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
	def forward(self, x):
		fea = self.head(x)
		body = self.body_tail(self.body(fea))
		fea = fea + body
		out = self.tail(fea)
		return out

# ---------- 判别器（可复用） ----------
class HSIPatchDiscriminator3D(nn.Module):
	def __init__(self, in_ch=1, base_nf=32, num_stages=4):
		super().__init__()
		layers = []
		def block(cin, cout, k=3, s=(1,2,2), p=1, norm=True):
			seq = [nn.Conv3d(cin, cout, k, s, p)]
			if norm:
				seq.append(nn.InstanceNorm3d(cout, affine=True))
			seq.append(nn.LeakyReLU(0.2, inplace=True))
			return nn.Sequential(*seq)
		c = base_nf
		layers.append(block(in_ch, c, norm=False, s=(1,1,1)))
		for _ in range(num_stages - 1):
			layers.append(block(c, c*2))
			c *= 2
		self.features = nn.Sequential(*layers)
		self.out_conv = nn.Conv3d(c, 1, kernel_size=3, stride=1, padding=1)
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
	def forward(self, x):
		f = self.features(x)
		out = self.out_conv(f)
		return out
