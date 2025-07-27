import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.empty(self.out_features, self.in_features))
        self._init_weight()
    
    def _init_weight(self):
        mean = 0
        std = math.sqrt(2.0 / (self.weight.size(0) + self.weight.size(1)))
        nn.init.trunc_normal_(self.weight, mean, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self._init_weight()

    def _init_weight(self):
        mean = 0
        std = math.sqrt(2.0 / (self.weight.size(0) + self.weight.size(1)))
        nn.init.trunc_normal_(self.weight, mean, std, -3 * std, 3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
        batch_size, seq_len = token_ids.shape[:2]
        embeddings = torch.empty(batch_size, seq_len, self.embedding_dim)
        for i, seq in enumerate(token_ids):
            for j, token_id in enumerate(seq):
                embeddings[i][j] = self.weight[token_id]
        return embeddings

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        norm_x = x / rms * self.gain
        return norm_x.to(self.dtype)

class RotaryPosisionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super(RotaryPosisionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # caculate and cache cos/sin theta
        i = torch.arange(max_seq_len).float()
        k = torch.arange(self.d_k // 2).float()
        theta_k = self.theta ** (2 * k / self.d_k)
        freqs = i.unsqueeze(1) / theta_k
        self.register_buffer('cos', torch.cos(freqs))
        self.register_buffer('sin', torch.sin(freqs))
                                                
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        rope_even = x_even * cos - x_odd * sin
        rope_odd = x_odd * cos + x_even * sin

        rope = torch.zeros_like(x)
        rope[..., ::2] = rope_even
        rope[..., 1::2] = rope_odd

        return rope
