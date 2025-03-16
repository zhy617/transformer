from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, d_model, heads, d_ff, max_seq_len, num_layers, device, drop_prob=0.1):
        """
        class for Decoder
        Args:
            dec_voc_size: Decoder vocabulary size
            d_model: Dimension of model
            heads: Number of heads
            d_ff: Dimension of feed forward
            max_seq_len: Maximum length of sequence
            num_layers: Number of layers
            device: Device (cpu or cuda)
            drop_prob: Dropout probability
        """
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size=dec_voc_size,
                                              embed_size=d_model,
                                              max_len=max_seq_len,
                                              drop_prob=drop_prob,
                                              device=device)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  heads=heads,
                                                  d_ff=d_ff,
                                                  drop_prob=drop_prob)
                                     for _ in range(num_layers)])
        
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, x, enc_src, trg_mask, src_mask):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer.forward(x, enc_src, trg_mask, src_mask)
        
        x = self.linear(x)
        # Todo: Add softmax
        return x