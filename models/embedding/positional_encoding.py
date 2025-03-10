import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device):
        """
        constructor of sinusoid positional encoding.
        a matrix of shape (max_len, d_model) 

        Args:
            d_model (int): dimension of model.
            max_len (int): max length of input sequence.
            device (torch.device): device - 'cuda' or 'cpu'.
        """
        super().__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        # don't need to calculate gradient
        self.encoding.requires_grad = False

        position = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        # 1D -> 2D
        
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 1D -> 2D
        # represents wavelength of 2pi ~ 2pi * 10000

        self.encoding[:, 0::2] = torch.sin(position / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(position / (10000 ** (_2i / d_model)))

    def forward(self, x: torch.Tensor):
        """
        Get positional encoding.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: positional encoding of input tensor.
        """
        return self.encoding[:x.size(1), :]
