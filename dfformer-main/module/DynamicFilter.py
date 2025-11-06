import torch
import torch.nn as nn
from torchinfo import summary
from torch.nn.modules.utils import _pair as to_2tuple

# -----------------------------
# 占位 Mlp 模块
# -----------------------------
class Mlp(nn.Module):
    """简单的 MLP 用于 reweight"""
    def __init__(self, in_features, expansion_ratio=0.25, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features * expansion_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# -----------------------------
# 占位 resize_complex_weight
# -----------------------------
def resize_complex_weight(weight, H, W):
    """占位函数，实际可用 F.interpolate 或其他方式调整大小"""
    # weight: [size, filter_size, num_filters, 2]
    # 为简单演示，直接重复或截取
    return weight

# -----------------------------
# StarReLU 定义
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
# DynamicFilter 模块
# -----------------------------
class DynamicFilter(nn.Module):
    """
    DynamicFilter 模块：
    --------------------
    作用：
    1. 对输入特征进行可学习的频域滤波，每个滤波器有自己的复数权重。
    2. 使用 MLP 生成每个滤波器的 reweight 系数，实现动态选择。
    3. 支持 act1 和 act2 自定义激活函数。

    输入: x [B, H, W, C]
    输出: x [B, H, W, C]
    """
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=0.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=False,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize

        # 通道扩展
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer() if callable(act1_layer) else act1_layer

        # 生成每个滤波器的动态权重
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)

        # 可学习的复数权重
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2) * 0.02
        )

        self.act2 = act2_layer() if callable(act2_layer) else act2_layer
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, _ = x.shape

        # 动态生成 reweight 系数
        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters, -1).softmax(dim=1)

        # 通道扩展 + 激活
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)

        # 二维频域变换
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # 获取复数权重
        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1], x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)

        routeing = routeing.to(torch.complex64)
        # 将多个滤波器加权
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)

        # 形状调整
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)

        # 频域加权
        x = x * weight
        # 逆傅里叶变换回空间域
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        # 空间域激活
        x = self.act2(x)
        # 通道投影回原始
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

    # 创建 DynamicFilter 模型
    model = DynamicFilter(dim=C, num_filters=4, size=H).to(device)

    # 前向传播
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    # torchinfo 查看结构
    summary(model, input_size=(B, H, W, C), device=device)
