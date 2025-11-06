import torch.nn as nn
import torch
from torchsummary import summary

class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_permute = pre_permute
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        if self.pre_permute:
            x = x.permute(0, 3, 1, 2)

        # 先进行 pre_norm
        x = self.pre_norm(x)
        x = self.conv(x)

        # 卷积后 -> [B, C, H, W]
        x = self.post_norm(x)
        # 转换回 [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        return x


# ------------------------------
# 测试脚本
# ------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型和输入都放到同一设备
    model = Downsampling(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=2,
        padding=1,
        pre_norm=nn.BatchNorm2d,
        post_norm=nn.BatchNorm2d,
        pre_permute=True
    ).to(device)

    x = torch.randn(2, 64, 64, 3).to(device)

    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    # torchsummary 也需要在相同 device 上运行
    summary(model, (64, 64, 3), device=str(device))
