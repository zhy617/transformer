import math

from torch import nn
import torch
from torch import Tensor

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor=None, e=1e-12):
        """
        Args:
            q: Query tensor with shape (batch_size, num_heads, seq_len_q, d_k)
            k: Key tensor with shape (batch_size, num_heads, seq_len_k, d_k)
            v: Value tensor with shape (batch_size, num_heads, seq_len_v, d_v)
            mask: Mask tensor with shape (batch_size, num_heads, seq_len_q, seq_len_k)
            e: Small value to avoid division by zero

        Returns:
            output: Output tensor with shape (batch_size, num_heads, seq_len_q, d_v)
            
            attention: Attention tensor with shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        d_k = q.size(-1)

        # Query * Key^T
        k_t = k.transpose(-2, -1)
        # scores with shape (batch_size, num_heads, seq_len_q, seq_len_k)
        scores :torch.Tensor = (q @ k_t) / math.sqrt(d_k)

        # Apply mask
        # src_mask: (batch_size, 1, 1, seq_len_k)
        # padding = 0 时 value = -10000
        # trg_mask: 
        print(scores.shape)
        print(mask.shape)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -10000)
        
        # Softmax
        scores = self.softmax(scores)

        # Attention Value
        output = scores @ v

        return output, scores