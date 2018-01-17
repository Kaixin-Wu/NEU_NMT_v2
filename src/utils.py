import math
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset

def get_threshold(n_in, n_out=None):

    if n_out:
        return math.sqrt(6. / (n_in + n_out))
    return math.sqrt(6. / n_in)

def load_dataset(batch_size):
    '''
    	load data sets.
    '''

    Lang1 = Field(include_lengths=True, init_token='<sos>', eos_token='<eos>')
    Lang2 = Field(include_lengths=True, init_token='<sos>', eos_token='<eos>')

    train = TranslationDataset(path='data/40w/train', exts=('.ch', '.en'), fields=(Lang1, Lang2))
    val = TranslationDataset(path='data/40w/valid', exts=('.ch', '.en'), fields=(Lang1, Lang2))
    test = TranslationDataset(path='data/40w/test', exts=('.ch', '.en'), fields=(Lang1, Lang2))

    Lang1.build_vocab(train.src, max_size=30000)
    Lang2.build_vocab(train.trg, max_size=30000)

    train_iter, val_iter, test_iter = BucketIterator.splits((train, val, test),
                                                            batch_size=batch_size, repeat=False)

    return train_iter, val_iter, test_iter, Lang1, Lang2
