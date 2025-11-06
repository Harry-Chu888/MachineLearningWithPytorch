import torch
import torch.nn as nn
from torchsummary import summary

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modified for [B, H, W, C] input
    """

    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False
        )

    def forward(self, x):
        # [B, H, W, C] -> [B, C, H, W]
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        # [B, C, H, W] -> [B, H, W, C]
        y = y.permute(0, 2, 3, 1)
        return y - x

# ------------------------------
# 测试脚本
# ------------------------------
if __name__ == "__main__":
    # 创建输入 [B, H, W, C]
    x = torch.randn(2, 32, 32, 64)  # batch=2, H=W=32, channels=64
    model = Pooling(pool_size=3)

    # 前向
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
