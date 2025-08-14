import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 基础积木 ----------

class SpectralSE(nn.Module):
    """
    光谱注意力（SE on spectral depth）。
    输入: [B, C_feat, D(C_bands), H, W]
    仅沿 H,W 池化，保留 D 维信息，再做 1x1x1 通道重标定。
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # 保留D, 池化H,W
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        w = self.avg_pool(x)  # [B, C, D, 1, 1]
        w = self.fc(w)        # [B, C, D, 1, 1]
        return x * w


class RRDB3D(nn.Module):
    """
    残差中的残差（简化版），全3D卷积，适合 HSI 的空间-光谱建模。
    """
    def __init__(self, nf=64, growth=32, use_spectral_se=True):
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
        out = x + 0.2 * self.se(c3)  # 稳定训练的缩放残差
        return out


class PixelShuffle2DIn3D(nn.Module):
    """
    仅在空间维(H,W)上做像素重排上采样，保持光谱深度D不变。
    输入: [B, C*r*r, D, H, W] -> 输出: [B, C, D, H*r, W*r]
    """
    def __init__(self, scale):
        super().__init__()
        assert scale in [2, 4], "支持 2x 或 4x（可叠两次2x实现4x）"
        self.scale = scale

    def forward(self, x):
        b, c, d, h, w = x.shape
        r = self.scale
        assert c % (r * r) == 0, "通道数需能被 r^2 整除"
        c_out = c // (r * r)
        x = x.view(b, c_out, r, r, d, h, w)
        x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()  # [B, C, D, H, r, W, r]
        x = x.view(b, c_out, d, h * r, w * r)
        return x


# ---------- 生成器 ----------

class HSIGenerator(nn.Module):
    """
    3D GAN 生成器: 3D卷积干路 + RRDB3D 堆叠 + 仅空间上采样 + 输出1通道(保留D光谱维)
    输入/输出: [B, 1, C_bands, H, W]  -> 超分后 [B, 1, C_bands, H*scale, W*scale]
    """
    def __init__(self, in_ch=1, base_nf=64, num_blocks=8, growth=32, scale=2, use_spectral_se=True):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv3d(in_ch, base_nf, 3, 1, 1)

        blocks = [RRDB3D(nf=base_nf, growth=growth, use_spectral_se=use_spectral_se)
                  for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)
        self.body_tail = nn.Conv3d(base_nf, base_nf, 3, 1, 1)

        # 上采样：仅在H,W上
        up_layers = []
        if scale == 4:
            # 两次2x
            for _ in range(2):
                up_layers += [
                    nn.Conv3d(base_nf, base_nf * 4, 3, 1, 1),
                    PixelShuffle2DIn3D(scale=2),
                    nn.LeakyReLU(0.2, inplace=True)
                ]
        elif scale == 2:
            up_layers += [
                nn.Conv3d(base_nf, base_nf * 4, 3, 1, 1),
                PixelShuffle2DIn3D(scale=2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        else:
            # 等比1x，做对比或消融
            up_layers += []

        self.upsampler = nn.Sequential(*up_layers)

        self.tail = nn.Sequential(
            nn.Conv3d(base_nf, base_nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_nf, 1, 3, 1, 1)
        )

        # Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 1, D, H, W]
        fea = self.head(x)
        body = self.body_tail(self.body(fea))
        fea = fea + body  # 长残差
        fea = self.upsampler(fea)
        out = self.tail(fea)  # [B, 1, D, H*, W*]
        return out


# ---------- 判别器 ----------

class HSIPatchDiscriminator3D(nn.Module):
    """
    3D PatchGAN：仅在空间维降采样(stride=(1,2,2))，保留光谱维D，
    末端对局部patch做判别（映射到 [B, 1, D', H', W'] 的实/假得分图）。
    """
    def __init__(self, in_ch=1, base_nf=64, num_stages=5):
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

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 1, D, H, W]
        f = self.features(x)
        out = self.out_conv(f)  # 判别图: [B, 1, D', H', W']
        return out