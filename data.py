from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()
data_loader = DataLoader(ext=('.en', '.de'), 
                        tokenize_en=tokenizer.tokenize_en, 
                        tokenize_de=tokenizer.tokenize_de,
                        init_token='<sos>',
                        eos_token='<eos>')

# first download the dataset and then make the dataset
train_data, valid_data, test_data = data_loader.make_dataset()

# build the vocabulary for the dataset
data_loader.build_vocab(train_data, min_freq=2)

# make the iterator for the dataset
train_iter, valid_iter, test_iter = data_loader.make_iter(train_data, valid_data,
                                                          test_data, batch_size, device)

# in the vocabulary, '<pad>' is used for padding, '<sos>' for start of sentence
# stoi: string to index
# itos: index to string
# src_pad_idx: integer value of '<pad>'
# trg_pad_idx: integer value of '<pad>'
# trg_sos_idx: integer value of '<sos>'
src_pad_idx = data_loader.source.vocab.stoi['<pad>']
trg_pad_idx = data_loader.target.vocab.stoi['<pad>']
trg_sos_idx = data_loader.target.vocab.stoi['<sos>']

# set the encoder and decoder vocabulary size
enc_voc_size = len(data_loader.source.vocab)
dec_voc_size = len(data_loader.target.vocab)