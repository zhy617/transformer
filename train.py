from data import *
from torch import nn
from torch.optim import Adam
import torch.optim as optim
from models.model.transformer import Transformer

def count_parameters(model):
    """
    Count the number of parameters with grad in the model
    Args:
        model: model to count the number of parameters
    Returns:
        int: number of parameters with grad in the model
    """
    # p.numel: return the number of elements in the tensor
    return sum(p.numel for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    """
    Initialize the weights of the model
    Args:
        m: model to initialize the weights
    """
    # If the model has weight and the dimension of the weight is greater than 1
    # except for ReLU, Bias
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    d_model=d_model,
                    d_ff=ffn_hidden,
                    n_heads=n_heads,
                    max_len=max_len,
                    n_layers=n_layers,
                    drop_prob=dropout,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')

model.apply(initialize_weights)

optimizer = Adam(model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

