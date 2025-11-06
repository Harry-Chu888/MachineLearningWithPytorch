import torch
import torch.nn as nn
from torchinfo import summary  # torchsummary 升级版，更稳定
from torch.nn.modules.utils import _pair as to_2tuple  # 用于将 size 转为 tuple

# -----------------------------
# StarReLU 定义（供 GlobalFilter 使用）
# -----------------------------
class StarReLU(nn.Module):
    """StarReLU: s * relu(x)^2 + b"""
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

# -----------------------------
# GlobalFilter 模块定义
# -----------------------------
class GlobalFilter(nn.Module):
    """
    GlobalFilter 模块：
    -------------------
    作用：
    1. 对输入特征进行全局频域滤波（RFFT + IRFFT）。
    2. 类似于一种频域注意力机制，通过可学习的复数权重调整每个频率分量。
    3. 在空间域中使用线性层和非线性激活增强表达能力。

    输入：
      x: Tensor of shape [B, H, W, C]
    输出：
      Tensor of shape [B, H, W, C]
    """
    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, size=14, **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1  # rfft2 后的频域宽度
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)

        # 逐通道线性映射扩展
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()

        # 可学习的复数频域权重
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, self.med_channels, 2, dtype=torch.float32) * 0.02
        )

        # 空间域非线性激活
        self.act2 = act2_layer()
        # 投影回原始通道数
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, _ = x.shape

        # 通道扩展
        x = self.pwconv1(x)
        x = self.act1(x)

        # 确保类型为 float32
        x = x.to(torch.float32)

        # 二维频域变换
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # 将权重转换为复数
        complex_weights = torch.view_as_complex(self.complex_weights)

        # 频域加权
        x = x * complex_weights

        # 逆傅里叶变换回空间域
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        # 空间域激活
        x = self.act2(x)

        # 投影回原始通道
        x = self.pwconv2(x)
        return x


# -----------------------------
# 测试脚本
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输入大小 B=2, H=W=14, C=64
    B, H, W, C = 2, 14, 14, 64
    x = torch.randn(B, H, W, C).to(device)

    # 创建 GlobalFilter 模型
    model = GlobalFilter(dim=C, size=H).to(device)

    # 前向传播测试
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"输出张量示例:\n{y[0,0,:5]}")  # 打印第一个位置前5个通道值

    # -----------------------------
    # 使用 torchinfo 查看模型结构
    # -----------------------------
    summary(model, input_size=(B, H, W, C), device=device)
