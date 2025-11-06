import torch.nn as nn

from Transformer.EncoderLayer import EncoderLayer
from Transformer.NormLayer import NormLayer
from Transformer.PositionalEncoder import PositionalEncoder


class Encoder(nn.Module):

    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(N)])
        self.norm = NormLayer(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
