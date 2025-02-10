

import torch
import torch.nn as nn
from .sparse_transformer_modules import Block

class CrossAttention(torch.nn.Module):

    def __init__(self, in_channels = 384, num_heads = 6, num_attention_blocks = 3, pool_method = "cls_token"):

        super().__init__()

        self.blocks = torch.nn.ModuleList([
            Block(in_channels, num_heads) for _ in range(num_attention_blocks)
        ])

        if pool_method.lower() not in ["max_pool", "avg_pool", "cls_token_pool"]:
            raise ValueError(f"Invalid pooling method: {pool_method}. Must be one of 'max_pool', 'avg_pool' or 'cls_token'.")
    
        self.pool_func = getattr(self, pool_method.lower())

    def max_pool(self, x):
        return x.max(dim=1).values

    def avg_pool(self, x):
        return x.mean(dim=1)
    
    def cls_token_pool(self, x):
        return x[:, 0, :]

    def forward(self, x, mask):
        
        #x: B x N x F tensor : batch of encoded parts, the classification token is the FIRST element in the N dimension
        #mask: B x M: boolean mask signifying which parts are real and which are padding

        B, N, F = x.shape

        for b in self.blocks:
            x = b(x, mask)

        #------------------------------------------------
        #Feature aggregation (classification token pooling)
        #B x N x F -> B x F global shape feature vector
        x = self.pool_func(x)
        #------------------------------------------------

        #------------------------------------------------
        #Feature aggregation (max pooling)
        #B x N x F -> B x F global shape feature vector
        #----x = x.max(dim=1).values
        #------------------------------------------------

        #------------------------------------------------
        #Feature aggregation (average pooling)
        #B x N x F -> B x F global shape feature vector
        #----x = x.mean(dim=1).values
        #------------------------------------------------

        return x

if __name__ == "__main__":

    pass