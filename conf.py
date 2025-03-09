import torch

# GPU device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model parameters setting
batch_size = 128 # one time a batch
max_len = 256 # max length of the input sequence
d_model = 512 # dimension of the model
n_layers = 6 # number of layers in the encoder and decoder
n_heads = 8 # number of heads in the multi-head attention
ffn_hidden = 2048 # hidden layer size in feed forward network
dropout = 0.1 # dropout rate

# optimizer parameters setting
init_lr = 1e-5 # initial learning rate
factor = 0.9 # factor for reducing learning rate
adam_eps = 5e-9 # eps in Adam optimizer
patience = 10 # patience for learning rate scheduler
warmup = 100 # warmup steps for learning rate scheduler
epoch = 100 # number of epochs
clip = 1.0 # gradient clipping
weight_decay = 5e-4 # weight decay rate
inf = float('inf') # infinity
