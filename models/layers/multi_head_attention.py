from torch import nn
from torch import Tensor
from models.layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        class for Multi-Head Attention
        Args:
            d_model: Dimension of model
            num_heads: Number of heads
        """
        super().__init__()
        self.n_head = num_heads
        self.attention = ScaleDotProductAttention()
        
        # every attention head has its own linear transformation
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask=None):
        """
        Forward pass of Multi-Head Attention
        Args:
            q: Query tensor with shape (batch_size, seq_len_q, d_model)
            k: Key tensor with shape (batch_size, seq_len_k, d_model)
            v: Value tensor with shape (batch_size, seq_len_v, d_model)
            mask: Mask tensor with shape (batch_size, seq_len_q, seq_len_k)
        Returns:
            (output, attention): Output tensor with shape (batch_size, seq_len_q, d_model), 
            Attention tensor with shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = q.size(0)
        
        # Linear transformation
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        # Split the tensor to multiple heads
        # q with shape (batch_size, seq_len_q, d_model) 
        # -> q with shape (batch_size, num_heads, seq_len_q, d_tensor)
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)
        
        # Apply attention
        # output with shape (batch_size, num_heads, seq_len_q, d_tensor)
        # attention with shape (batch_size, num_heads, seq_len_q, seq_len_k)
        output, attention = self.attention.forward(q, k, v, mask)

        # Concat

    def split(self, tensor:Tensor):
        """
        split the tensor to multiple heads
        Args:
            tensor: Input tensor with shape (batch_size, length, d_model)
        Returns:
            tensor: Output tensor with shape (batch_size, num_heads, length, d_tensor)
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor
    
    def concat(self, tensor:Tensor):
        """
        Concatenate the tensor
        Args:
            tensor: Input tensor with shape (batch_size, num_heads, length, d_tensor)
        Returns:
            tensor: Output tensor with shape (batch_size, length, d_model)
        """
        batch_size, num_heads, length, d_tensor = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_tensor * num_heads)
        return tensor
