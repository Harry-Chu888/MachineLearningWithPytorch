import torch
import torch.nn as nn
from torchinfo import summary

class RandomMixing(nn.Module):
    """
    RandomMixing 模块：
    ------------------
    这个模块实现了对输入特征图的**随机线性混合**操作，类似于一种固定的 token 交互方式。
    它的作用可以总结为：
      1. 将输入的每个空间位置（token）重新与其他位置进行线性组合，
         通过一个固定的随机权重矩阵 random_matrix。
      2. 模块不参与训练（random_matrix requires_grad=False），
         所以混合模式在训练过程中保持不变。
      3. 可以增强特征之间的全局交互，类似于轻量的自注意力机制，但不需要计算注意力权重。

    输入:
      x: Tensor of shape [B, H, W, C], B=batch, HxW=token数, C=通道数
    输出:
      Tensor of shape [B, H, W, C]，经过随机混合后的特征
    """
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1),
            requires_grad=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 假设输入大小 H=W=14, num_tokens=H*W=196
    B, H, W, C = 2, 14, 14, 64
    num_tokens = H * W

    # 创建模型
    model = RandomMixing(num_tokens=num_tokens).to(device)

    # 构造输入
    x = torch.randn(B, H, W, C).to(device)

    # 前向传播测试
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"输出张量示例:\n{y[0,0,:5]}")  # 打印第一个位置前5个通道值

    # -----------------------------
    # 使用 torchinfo 查看模型结构
    # -----------------------------
    summary(model, input_size=(B, H, W, C), device=device)
