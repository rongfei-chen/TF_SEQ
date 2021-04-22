import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from TransformerModel import TransformerModel


class LoadAndBatchData:
    def __init__(self, batch_size, eval_batch_size, bptt, emsize, nhid, nlayers, nhead, dropout):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_iter = WikiText2(split='train')
        self.tokenizer = get_tokenizer('basic_english')
        self.counter = Counter()
        self.batch_size = batch_size  # batch_size = 20
        self.eval_batch_size = eval_batch_size  # eval_batch_size = 10
        self.bptt = bptt  # bptt=35

        self.emsize = emsize  # embedding dimension
        self.nhid = nhid  # the dimension of the feedforward network model in nn.TransformerEncoder
        self.nlayers = nlayers  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.nhead = nhead  # the number of heads in the multi-head attention models
        self.dropout = dropout  # the dropout value

    def getVocab(self):
        for line in self.train_iter:
            self.counter.update(self.tokenizer(line))
        vocab = Vocab(self.counter)
        return vocab

    # 这里的train_iter与self.train_iter是不同的
    def data_process(self, raw_text_iter):  # 需输入参数train_iter
        vocab = self.getVocab()
        data = [torch.tensor([vocab[token] for token in self.tokenizer(item)],
                             dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data, bsz):
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(self.device)

    # 使用时需声明 train_iter, val_iter, test_iter = WikiText2()
    def getData(self, train_iter, val_iter, test_iter):
        data_train = self.data_process(train_iter)
        data_val = self.data_process(val_iter)
        data_test = self.data_process(test_iter)
        train_data = self.batchify(data_train,self.batch_size)
        val_data = self.batchify(data_val,self.eval_batch_size)
        test_data = self.batchify(data_test,self.eval_batch_size)
        return train_data, val_data, test_data

    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].reshape(-1)
        return data, target

    def initInstance(self):
        n_tokens = len(self.getVocab())  # the size of vocabulary
        model = TransformerModel(n_tokens, self.emsize, self.nhead, self.nhid, self.nlayers, self.dropout).to(self.device)
        return model, n_tokens















# train_iter, val_iter, test_iter = WikiText2() 后续输入需要

