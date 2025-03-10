from torch import nn

class TokenEmbedding(nn.Embedding):
    """
    input: (batch_size, seq_len) , each element is the index of token in vocabulary
    output: (batch_size, seq_len, embed_size) , each element is the embedding of token
    """
    def __init__(self, vocab_size: int, embed_size: int):
        """
        Initialize token embedding.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Size of the embedding.
        """
        super().__init__(vocab_size, embed_size, padding_idx=1)

        