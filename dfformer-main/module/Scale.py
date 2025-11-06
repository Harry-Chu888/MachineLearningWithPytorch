import torch.nn as nn
import torch
from torchinfo import summary


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 创建模型并放到设备上
    model = Scale(dim=4, init_value=0.5).to(device)

    # 构造输入张量并放到同一设备
    x = torch.randn(2, 4).to(device)  # batch=2, dim=4

    # 前向传播测试
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"输出张量:\n{y}")

    # -----------------------
    # 使用 torchsummary 查看模型结构
    # -----------------------
    # 对于线性向量输入，input_size 应该是 (dim,)
    summary(model, input_size=(4,), device=str(device))
