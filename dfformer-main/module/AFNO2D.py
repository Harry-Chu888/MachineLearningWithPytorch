import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple

# -----------------------------
# StarReLU 激活
# -----------------------------
from torchsummary import summary


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
# AFNO2D 模块
# -----------------------------
# 作用：
# AFNO2D 是一种基于傅里叶变换的全局特征提取模块。
# 它利用二维快速傅里叶变换（FFT）将输入特征映射到频域，
# 在频域上进行线性变换（类似全连接层，但在频域），
# 之后再通过逆傅里叶变换（iFFT）映射回空间域。
#
# 核心特点：
# 1. 全局感受野：因为 FFT 是全局运算，AFNO2D 可以捕获图像或特征图中远距离的依赖关系。
# 2. 高效稀疏操作：通过“hard thresholding”只保留频域的主要模式，降低计算复杂度。
# 3. 可学习的频域权重：在频域中每个块（block）有可学习参数，用于调节不同频率的响应。
# 4. 类似 Transformer 的处理方式：先做归一化+激活，再频域操作，最后残差连接。
#
# 典型输入输出：
# 输入 x: [B, H, W, C] 形式的特征图
# 输出 x: [B, H, W, C] 形式的特征图，包含经过频域增强后的全局信息
#
# 适用场景：
# 适合用于视觉任务中的特征提取模块，如图像分类、语义分割、目标检测等，
# 特别是希望捕获远距离依赖关系时。
class AFNO2D(nn.Module):
    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, size=14,
                 num_blocks=8, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1, hidden_size_factor=1,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)

        assert self.med_channels % num_blocks == 0, \
            f"hidden_size {self.med_channels} should be divisible by num_blocks {num_blocks}"

        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.med_channels // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        # 通道扩展
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer() if callable(act1_layer) else act1_layer

        # 频域参数
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size,
                                                        self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks,
                                                        self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.act2 = act2_layer() if callable(act2_layer) else act2_layer
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        bias = x.clone()

        # FFT
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x_fft = x_fft.reshape(B, self.size, self.filter_size, self.num_blocks, self.block_size)

        kept_modes = int(self.filter_size * self.hard_thresholding_fraction)

        # 输出初始化
        o1_real = torch.zeros([B, self.size, self.filter_size, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros_like(x_fft)
        o2_imag = torch.zeros_like(x_fft)

        # 第一个线性 + ReLU
        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x_fft[:, :, :kept_modes].real, self.w1[0]) -
            torch.einsum('...bi,bio->...bo', x_fft[:, :, :kept_modes].imag, self.w1[1]) +
            self.b1[0]
        )
        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x_fft[:, :, :kept_modes].imag, self.w1[0]) +
            torch.einsum('...bi,bio->...bo', x_fft[:, :, :kept_modes].real, self.w1[1]) +
            self.b1[1]
        )

        # 第二个线性
        o2_real[:, :, :kept_modes] = (
                torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) -
                torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) +
                self.b2[0]
        )
        o2_imag[:, :, :kept_modes] = (
                torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) +
                torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) +
                self.b2[1]
        )

        # Softshrink
        o2_real = F.softshrink(o2_real.float(), lambd=self.sparsity_threshold)
        o2_imag = F.softshrink(o2_imag.float(), lambd=self.sparsity_threshold)
        x_fft = torch.complex(o2_real, o2_imag)

        # 直接 irfft2，恢复 [B,H,W,C]
        x = torch.fft.irfft2(x_fft, s=(H,W), dim=(1,2), norm='ortho')
        x = x + bias
        x = self.act2(x)
        x = self.pwconv2(x)
        return x

# -----------------------------
# 测试
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W, C = 2, 14, 14, 64
    x = torch.randn(B, H, W, C).to(device)

    model = AFNO2D(dim=C, expansion_ratio=2, num_blocks=4, size=H).to(device)
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    summary(model, (B, H, W, C))
