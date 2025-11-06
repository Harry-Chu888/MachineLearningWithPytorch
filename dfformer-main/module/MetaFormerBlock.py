import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, drop=0., bias=False):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim))
    def forward(self, x):
        return x * self.scale

# ------------------------------
# MetaFormerBlock
# ------------------------------
class MetaFormerBlock(nn.Module):
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp_layer=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None,
                 size=14):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer() if token_mixer == nn.Identity else token_mixer(dim=dim)
        self.drop_path1 = nn.Identity(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(dim=dim, drop=drop)
        self.drop_path2 = nn.Identity(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x

# ------------------------------
# 测试脚本
# ------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入 [B, N, C]，假设 N=196, C=64
    x = torch.randn(2, 196, 64).to(device)

    model = MetaFormerBlock(dim=64).to(device)

    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
