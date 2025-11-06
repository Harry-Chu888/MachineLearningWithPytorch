import torch
import torch.nn as nn
from torchinfo import summary

class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = SquaredReLU().to(device)

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
