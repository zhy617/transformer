import math

from torch import nn

class ScaleDotProductAttention(nn.Module):
    def __init__f(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None, e=1e-12):
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
        scores = (q @ k_t) / math.sqrt(d_k)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -10000)
        
        # Softmax
        scores = self.softmax(scores)

        # Attention Value
        output = scores @ v

        return output, scores