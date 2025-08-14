# FCNN3D.py
import torch
import torch.nn as nn

class FCNN3D(nn.Module):

    """
    3D Fully Convolutional Network for Hyperspectral Image Super-Resolution
    输入:  [B, 1, C, H, W]  # B: batch, C: 波段数, H: 高度, W: 宽度
    输出:  [B, 1, C', H', W']  # C', H', W' 根据卷积核和padding变化
    """

    def  __init__(self, in_channels=1):
        super(FCNN3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(7, 9, 9), padding=(3, 4, 4))
        self.conv2 = nn.Conv3d(64, 32, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.conv3 = nn.Conv3d(32, 9, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.conv4 = nn.Conv3d(9, 1, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

def center_crop(tensor, target_size):
    """
    中心裁剪 3D 张量
    tensor: [B, C, D, H, W]
    target_size: (H_target, W_target)
    """
    _, _, D, H, W = tensor.shape
    dy, dx = (H - target_size[0]) // 2, (W - target_size[1]) // 2
    return tensor[..., dy:H-dy, dx:W-dx]

if __name__ == "__main__":
    model = FCNN3D(in_channels=1)
    dummy_input = torch.randn(2, 1, 32, 64, 64)
    output = model(dummy_input)
    print("输入尺寸:", dummy_input.shape)
    print("输出尺寸:", output.shape)

    hr_target = torch.randn_like(output)
    cropped_out = center_crop(output, (hr_target.shape[-2], hr_target.shape[-1]))
    cropped_hr = center_crop(hr_target, (cropped_out.shape[-2], cropped_out.shape[-1]))
    loss_fn = nn.MSELoss()
    loss = loss_fn(cropped_out, cropped_hr)
    print("测试损失:", loss.item())