import numpy as np
import chainer
from chainer import Variable, Chain
from chainer import functions as F
from chainer import links     as L
from chainer import reporter  as R


class ONT(Chain):

    def __init__(self, n_cell, size_hidden, rate_dropout):
        super(ONT, self).__init__()
        self.rate_dropout = rate_dropout
        with self.init_scope():
            self.rnn_a = L.NStepRNNReLU(n_cell, 300, size_hidden, rate_dropout)
            self.rnn_b = L.NStepRNNReLU(n_cell, 300, size_hidden, rate_dropout)
            self.l1  = L.Highway(size_hidden*2)
            self.l2  = L.Linear(size_hidden*2, 4)

    def forward(self, xs_a, xs_b):
        hs_a, _ = self.rnn_a(None, xs_a)
        hs_b, _ = self.rnn_b(None, xs_b)
        hs = F.concat((hs_a[0], hs_b[0]))
        hs = F.dropout(hs, ratio=self.rate_dropout)
        hs = F.sigmoid(self.l1(hs))
        hs = F.dropout(hs, ratio=self.rate_dropout)
        hs = self.l2(hs)

        return hs



class ONT_LSTM(Chain):

    def __init__(self, n_cell, size_hidden, rate_dropout):
        super(ONT_LSTM, self).__init__()
        self.rate_dropout = rate_dropout
        with self.init_scope():
            self.rnn_a = L.NStepLSTM(n_cell, 300, size_hidden, rate_dropout)
            self.rnn_b = L.NStepLSTM(n_cell, 300, size_hidden, rate_dropout)
            self.l1  = L.Highway(size_hidden*2)
            self.l2  = L.Linear(size_hidden*2, 4)

    def forward(self, xs_a, xs_b):
        hs_a, _, _ = self.rnn_a(None, None, xs_a)
        hs_b, _, _ = self.rnn_b(None, None, xs_b)
        hs = F.concat((hs_a[0], hs_b[0]))
        hs = F.dropout(hs, ratio=self.rate_dropout)
        hs = F.sigmoid(self.l1(hs))
        hs = F.dropout(hs, ratio=self.rate_dropout)
        hs = self.l2(hs)

        return hs


class ONT_BiLSTM(Chain):

    def __init__(self, n_cell, size_hidden, rate_dropout):
        super(ONT_BiLSTM, self).__init__()
        self.rate_dropout = rate_dropout
        with self.init_scope():
            self.rnn_a = L.NStepBiLSTM(n_cell, 300, size_hidden, rate_dropout)
            self.rnn_b = L.NStepBiLSTM(n_cell, 300, size_hidden, rate_dropout)
            self.l1  = L.Linear(size_hidden*4, size_hidden*4)
            self.l2  = L.Linear(size_hidden*4, 4)

    def forward(self, xs_a, xs_b):
        hs_a, _, _ = self.rnn_a(None, None, xs_a)
        hs_b, _, _ = self.rnn_b(None, None, xs_b)
        hs = F.concat((hs_a[0], hs_a[1], hs_b[0], hs_b[1]))
        hs = F.dropout(hs, ratio=self.rate_dropout)
        hs = F.sigmoid(self.l1(hs))
        hs = F.dropout(hs, ratio=self.rate_dropout)
        hs = self.l2(hs)

        return hs


class ONT_GRU(Chain):

    def __init__(self, n_cell, size_hidden, rate_dropout):
        super(ONT_GRU, self).__init__()
        self.rate_dropout = rate_dropout
        with self.init_scope():
            self.rnn_a = L.NStepGRU(n_cell, 300, size_hidden, rate_dropout)
            self.rnn_b = L.NStepGRU(n_cell, 300, size_hidden, rate_dropout)
            self.l1  = L.Highway(size_hidden*2)
            self.l2  = L.Linear(size_hidden*2, 4)

    def forward(self, xs_a, xs_b):
        hs_a, _ = self.rnn_a(None, xs_a)
        hs_b, _ = self.rnn_b(None, xs_b)
        hs = F.concat((hs_a[0], hs_b[0]))
        hs = F.dropout(hs, ratio=self.rate_dropout)
        hs = F.sigmoid(self.l1(hs))
        hs = F.dropout(hs, ratio=self.rate_dropout)
        hs = self.l2(hs)

        return hs


class ONT_BiGRU(Chain):

    def __init__(self, n_cell, size_hidden, rate_dropout):
        super(ONT_BiGRU, self).__init__()
        self.rate_dropout = rate_dropout
        with self.init_scope():
            self.rnn_a = L.NStepBiGRU(n_cell, 300, size_hidden, rate_dropout)
            self.rnn_b = L.NStepBiGRU(n_cell, 300, size_hidden, rate_dropout)
            self.l1  = L.Linear(size_hidden*4, size_hidden*4)
            self.l2  = L.Linear(size_hidden*4, 4)

    def forward(self, xs_a, xs_b):
        hs_a, _ = self.rnn_a(None, xs_a)
        hs_b, _ = self.rnn_b(None, xs_b)
        hs = F.concat((hs_a[0], hs_a[1], hs_b[0], hs_b[1]))
        hs = F.dropout(hs, ratio=self.rate_dropout)
        hs = F.sigmoid(self.l1(hs))
        hs = F.dropout(hs, ratio=self.rate_dropout)
        hs = self.l2(hs)

        return hs
