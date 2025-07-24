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

