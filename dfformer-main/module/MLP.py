import torch
import torch.nn as nn
from torchsummary import summary

# 假设之前定义过的激活函数
class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(torch.ones(1) * scale_value, requires_grad=scale_learnable)
        self.bias = nn.Parameter(torch.ones(1) * bias_value, requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class SquaredReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))

# 工具函数
def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

# ------------------------------
# MLP 模块
# ------------------------------
class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# ------------------------------
# MLP Head
# ------------------------------
class MlpHead(nn.Module):
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
                 norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x

# ------------------------------
# 测试脚本
# ------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(2, 64).to(device)

    mlp_model = Mlp(dim=64).to(device)
    head_model = MlpHead(dim=64, num_classes=10).to(device)

    y = mlp_model(x)
    y_head = head_model(x)

    print(f"Mlp 输入: {x.shape}, 输出: {y.shape}")
    print(f"MlpHead 输入: {x.shape}, 输出: {y_head.shape}")

    print("\n--- Mlp summary ---")
    summary(mlp_model, input_size=(64,), device=str(device))

    print("\n--- MlpHead summary ---")
    summary(head_model, input_size=(64,), device=str(device))
