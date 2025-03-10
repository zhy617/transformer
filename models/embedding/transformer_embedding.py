from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, drop_prob, device):
        """
        class for word embedding that include positional encoding.
        Args:
            vocab_size (int): size of vocabulary.
            embed_size (int): size of embedding.
            max_len (int): max length of input sequence.
            drop_prob (float): probability of dropout.
            device (torch.device): device - 'cuda' or 'cpu'.
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len, device)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        """
        forward pass of transformer embedding.
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len).
        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_len, embed_size).
        """
        return self.dropout(self.token_embedding(x) + self.positional_encoding(x))