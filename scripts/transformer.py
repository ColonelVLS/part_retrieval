

import torch
import torch.nn as nn
from .transformer_modules import Mlp, Attention, Block
    
class CrossAttention(torch.nn.Module):

    def __init__(self, in_channels = 384, num_heads = 6, num_attention_blocks = 3, pool_method = "cls_token"):

        super().__init__()
        self.block = torch.nn.Sequential(
            *[Block(in_channels, num_heads) for _ in range(num_attention_blocks)]
        )

        if pool_method.lower() not in ["max_pool", "avg_pool", "cls_token_pool"]:
            raise ValueError(f"Invalid pooling method: {pool_method}. Must be one of 'max_pool', 'avg_pool' or 'cls_token'.")
    
        self.pool_func = getattr(self, pool_method.lower())

    def max_pool(self, x):
        return x.max(dim=2).values

    def avg_pool(self, x):
        return x.mean(dim=2)
    
    def cls_token_pool(self, x):
        return x[:, :, 0, :]

    def forward(self, x, y):
        
        B, N, F = x.shape
        #x: B x N x F tensor : batch of encoded parts
        #y: B x M x F tensor : batch of encoded warehouse parts

        #A new tensor B x M x (N+1) x F is created and self attention is performed,
        #between all parts in x and each individual part in y, added one at a time,
        #for each element in the batch. Finally, pooling is performed over N+1,
        #to obtain a single shape feature vector for each warehouse part we want to evaluate.

        assert x.dim() == 3 and y.dim() == 3, f"Input tensors must be 3-dimensional. Got x:{x.shape} and y:{y.shape}"
        assert x.shape[-1] == y.shape[-1], f"Feature dimension mismatch. Got x:{x.shape} and y:{y.shape}"

        #B x N x F -> repeat -> B x M x N x F
        x = x.unsqueeze(1).repeat(1, y.shape[1], 1, 1)
        #B x M x F -> permute -> B x M x 1 x F
        y = y.unsqueeze(2)

        #B x M x (N+1) x F -> B * M x (N+1) x F
        z = torch.cat([x, y], dim=2)
        z = z.reshape(-1, z.shape[2], z.shape[-1])
        z = self.block(z)

        #B * M x (N+1) x F -> B x M x (N+1) x F
        z = z.reshape(B, y.shape[1], -1, z.shape[-1])

        #------------------------------------------------
        #Feature aggregation
        #B x M x (N+1) x F -> B x M x F (shape feature vector, for each tried out warehouse part)
        z = self.pool_func(z)
        #------------------------------------------------

        return z

if __name__ == "__main__":

    import time

    model = CrossAttentionB3().cuda()
    x = torch.rand(1, 10, 384).cuda()
    y = torch.rand(1, 1, 384).cuda()

    z = model(x, y)