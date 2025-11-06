import torch
import torch.nn as nn
from torchinfo import summary

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = StarReLU(scale_value=1.0, bias_value=0.1).to(device)

    # 构造输入张量
    x = torch.randn(2, 3, 32, 32).to(device)  # 示例：batch=2, channels=3, H=W=32

    # 前向传播测试
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"输出张量示例:\n{y[0,0,:5,:5]}")  # 打印第一个通道前5x5部分

    # -----------------------------
    # 使用 torchinfo 查看模型结构
    # -----------------------------
    summary(model, input_size=(2, 3, 32, 32), device=device)
