from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedforward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, drop_prob=0.1):
        """
        class for Decoder
        Args:
            d_model: Dimension of model
            heads: Number of heads
            d_ff: Dimension of feed forward
            drop_prob: Dropout probability
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model, heads)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.feed_forward = PositionwiseFeedforward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        """
        forward pass of Decoder
        Args:
            dec: Decoder input tensor with shape (batch_size, trg_len, d_model)
            enc: Encoder output tensor with shape (batch_size, src_len, d_model)
            trg_mask: Target mask tensor with shape (batch_size, 1, trg_len, trg_len)
            src_mask: Source mask tensor with shape (batch_size, 1, 1, src_len)
        """
        # Self-Attention
        _x = dec
        x = self.self_attention.forward(dec, dec, dec, trg_mask)

        # Add & Norm
        x = self.dropout1(x)
        x = self.norm1.forward(x + _x)

        # Encoder-Decoder Attention
        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # Add & Norm
            x = self.dropout2(x)
            x = self.norm2.forward(x + _x)
        
        # Position-wise Feed Forward
        _x = x
        x = self.feed_forward.forward(x)

        # Add & Norm
        x = self.dropout3(x)
        x = self.norm3.forward(x + _x)

        return x


