import torch
import torch.nn as nn
from .modules import RMSNorm, Embedding, Linear
from .layers import MultiHeadSelfAttention, SwiGLUFFN

class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: int
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, True, max_seq_len, theta)
        self.ffn = SwiGLUFFN(d_model, d_ff)
    def forward(self, x):
        y = x + self.attention(self.rmsnorm1(x))
        output = y + self.ffn(self.rmsnorm2(y))
        return output
    
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ):
        super(TransformerLM, self).__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            Transformer(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        )
        self.rms_norm = RMSNorm(d_model)
        self.output_embedding = Linear(d_model, vocab_size)
    
    def forward(self, indices):
        x = self.token_embedding(indices)

        for layer in self.layers:
            x = layer(x)
        
        x_norm = self.rms_norm(x)
        output = self.output_embedding(x_norm)
        return output




