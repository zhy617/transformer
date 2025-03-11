from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedforward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, drop_prob=0.1):
        """
        class for Encoder Layer
        Args:
            d_model: Dimension of model
            heads: Number of heads
            d_ff: Dimension of feed forward
            drop_prob: Dropout probability
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, heads)
        self.dropout1 = nn.Dropout(drop_prob)
        self.norm1 = LayerNorm(d_model)

        self.feed_forward = PositionwiseFeedforward(d_model, d_ff)
        self.dropout2 = nn.Dropout(drop_prob)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, src_mask):
        """
        forward pass of Encoder Layer
        Args:
            x: Input tensor with shape (batch_size, seq_len, d_model)
            src_mask: Source mask tensor with shape (batch_size, 1, 1, seq_len)
        """
        # Multi-Head Attention
        _x = x
        x, _ = self.attention.forward(x, x, x, src_mask)

        # Add & Norm
        x = self.dropout1(x)
        x = self.norm1.forward(x + _x)

        # Position-wise Feed Forward
        _x = x
        x = self.feed_forward.forward(x)

        # Add & Norm
        x = self.dropout2(x)
        x = self.norm2.forward(x + _x)

        return x