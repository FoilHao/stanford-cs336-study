import torch
import torch.nn as nn

from .modules import Linear, Embedding

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
