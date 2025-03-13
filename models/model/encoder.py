from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, heads,
                 d_ff, n_layers, device, drop_prob=0.1):
        """
        class for Encoder
        Args:
            enc_voc_size: Encoder vocabulary size
            max_len: Maximum length of input sequence
            d_model: Dimension of model
            heads: Number of heads
            d_ff: Dimension of feed forward
            n_layers: Number of layers
            device: Device (cpu or cuda)
            drop_prob: Dropout probability
        """
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size=enc_voc_size,
                                              embed_size=d_model,
                                              max_len=max_len,
                                              drop_prob=drop_prob,
                                              device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  heads=heads,
                                                  d_ff=d_ff,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        
    def forward(self, x, src_mask):
        """
        forward pass of Encoder
        Args:
            x: Input tensor with shape (batch_size, seq_len)
            src_mask: Source mask tensor with shape (batch_size, 1, 1, seq_len)
        """
        x = self.embedding.forward(x)
        for layer in self.layers:
            x = layer.forward(x, src_mask)
        return x