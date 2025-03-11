import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        """
        class for Layer Normalization
        Args:
            d_model: Dimension of model
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        """
        Forward pass of Layer Normalization
        Args:
            x: Input tensor
        Returns:
            out: Normalized tensor
        """
        # keepdim=True: keep the dimension of input x
        mean = x.mean(-1, keepdim=True)
        # unbiased=False: calculate the biased variance ???
        var = x.var(-1, unbiased=False, keepdim=True)

        # normalize the input x
        # eps: small value to avoid division by zero
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
