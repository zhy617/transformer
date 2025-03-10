from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import Multi30k, TranslationDataset
ROOT = 'C:/Users/13183/OneDrive/PDSL/my_transformer/data/Multi30k/'

class DataLoader():
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        """
        parameters:
        ext: tuple, ('.de', '.en') or ('.en', '.de')
        tokenize_en method: tokenizer for English
        tokenize_de method: tokenizer for German
        sos_token: start of sentence token
        eos_token: end of sentence token
        """
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset start initializing...')
    
    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, 
                                eos_token=self.eos_token, lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, 
                                eos_token=self.eos_token, lower=True, batch_first=True)
        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, 
                                eos_token=self.eos_token, lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token,
                                eos_token=self.eos_token, lower=True, batch_first=True)
        
        # because of bad internet connection, I download the dataset manually
        (train_data, valid_data, test_data) = TranslationDataset.splits(exts=self.ext,
            path=ROOT, fields=(self.source, self.target), test='test2016')
        return train_data, valid_data, test_data
    
    def build_vocab(self, train_data, min_freq):
        """
        parameters:
        train_data: training dataset
        min_freq: acceptable minimum frequency of the word
        """
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train_data, valid_data, test_data, batch_size, device):

        # use BucketIterator to make batches, because the length 
        # of the sentences are different
        train_iter, valid_iter, test_iter = BucketIterator.splits(
            (train_data, valid_data, test_data), batch_size=batch_size, device=device)
        
        print('dataset initializing is done!')
        return train_iter, valid_iter, test_iter