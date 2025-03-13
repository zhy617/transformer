from torch import nn
import torch

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size,
                 d_model, d_ff, n_heads, max_len, n_layers, drop_prob, device):
        """
        class for Transformer model
        Args:
            src_pad_idx: padding index for source language
            trg_pad_idx: padding index for target language
            trg_sos_idx: start of sentence index for target language
            enc_voc_size: size of the source vocabulary
            dec_voc_size: size of the target vocabulary
            d_model: dimension of model
            d_ff: dimension of feed forward layer
            n_heads: number of heads
            max_len: maximum length of the sequence
            n_layers: number of layers
            drop_prob: dropout probability
            device: device to run the model
        """
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(enc_voc_size=enc_voc_size,
                               max_len=max_len,
                               d_model=d_model,
                               heads=n_heads,
                               d_ff=d_ff,
                               n_layers=n_layers,
                               drop_prob=drop_prob,
                               device=device)
        
        self.decoder = Decoder(dec_voc_size=dec_voc_size,
                               d_model=d_model,
                               heads=n_heads,
                               d_ff=d_ff,
                               max_seq_len=max_len,
                               num_layers=n_layers,
                               drop_prob=drop_prob,
                               device=device)
        
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, src_mask, trg_mask)
        return output

    def make_src_mask(self, src):
        """
        make mask for source sequence
        Args:
            src: source sequence with shape (batch_size, seq_len)
        Returns:
            mask: mask tensor with shape (batch_size, 1, 1, seq_len)
        """
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        """
        make mask for target sequence
        Args:
            trg: target sequence with shape (batch_size, seq_len)
        Returns:
            mask: mask tensor with shape (batch_size, 1, seq_len, seq_len)
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    