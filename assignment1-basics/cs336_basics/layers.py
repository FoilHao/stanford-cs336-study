import torch
import torch.nn as nn
from einops import einsum, rearrange

from .modules import Linear, Embedding, RotaryPosisionalEmbedding

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(SwiGLUFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    
    def SiLU(self, x):
        return x * torch.sigmoid(x)
    
    def forward(self, x):
        return self.w2(self.SiLU(self.w1(x)) * self.w3(x))

def softmax(x, dim):
    max_val = torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(x - max_val)
    sum_exp = torch.sum(exp, dim=dim, keepdim=True)
    return exp / sum_exp

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    product = Q @ K.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        product = product.masked_fill(~mask, float('-inf'))
    attention = softmax(product, -1) @ V
    return attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float | None = None,
        token_positions: torch.Tensor | None = None
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.token_positions = token_positions
        self.d_k = self.d_model // self.num_heads
        self.apply_rope = RotaryPosisionalEmbedding(theta, self.d_k, max_seq_len) if use_rope else None

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        
    def _create_causual_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask
    
    def forward(self, x):
        seq_len = x.shape[-2]
        qkv_proj = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
        qkv = x @ qkv_proj.T
        q, k, v = qkv.chunk(3, -1)

        q = rearrange(
            q, "... seq_len (h d_head) -> ... h seq_len d_head", h = self.num_heads
        )
        k = rearrange(
            k, "... seq_len (h d_head) -> ... h seq_len d_head", h = self.num_heads
        )
        v = rearrange(
            v, "... seq_len (h d_head) -> ... h seq_len d_head", h = self.num_heads
        )
        # tranpose head dim to batch dim
        if self.use_rope:
            q = self.apply_rope(q, self.token_positions)
            k = self.apply_rope(k, self.token_positions)
        
        casual_mask = self._create_causual_mask(seq_len)
        output = scaled_dot_product_attention(q, k, v, casual_mask)
        output = rearrange(
            output, "... h seq_len d_head -> ... seq_len (h d_head)"
        )
        return self.o_proj(output)

