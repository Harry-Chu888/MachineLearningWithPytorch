import torch
import torch.nn as nn
from torchinfo import summary

# -----------------------------
# StarReLU 定义（供 SepConv 使用）
# -----------------------------
class StarReLU(nn.Module):
    """StarReLU: s * relu(x)^2 + b"""
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(torch.tensor(scale_value), requires_grad=scale_learnable)
        self.bias = nn.Parameter(torch.tensor(bias_value), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

# -----------------------------
# SepConv 模块定义
# -----------------------------
class SepConv(nn.Module):
    """
    Inverted separable convolution from MobileNetV2:
    1x1 pointwise conv -> depthwise conv -> 1x1 pointwise conv
    支持自定义激活函数 act1 / act2
    """
    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, kernel_size=7, padding=3,
                 **kwargs):
        super().__init__()
        med_channels = int(expansion_ratio * dim)

        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer() if callable(act1_layer) else act1_layer

        # depthwise conv
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias
        )

        self.act2 = act2_layer() if callable(act2_layer) else act2_layer
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        # [B, H, W, C]
        x = self.pwconv1(x)
        x = self.act1(x)

        # 调整为 [B, C, H, W] 才能用 depthwise conv
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)

        # 回到 [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x

# -----------------------------
# 测试脚本
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输入 [B, H, W, C]
    B, H, W, C = 2, 14, 14, 64
    x = torch.randn(B, H, W, C).to(device)

    # 创建 SepConv 模型
    model = SepConv(dim=C, expansion_ratio=2, kernel_size=3, padding=1).to(device)

    # 前向传播
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    # 使用 torchinfo 打印模型结构
    summary(model, input_size=(B, H, W, C), device=device)
