import pickle
import numpy as np
import chainer
from chainer import Variable, Chain
from chainer import functions as F
from chainer import links     as L
from chainer import reporter  as R


class ONT_Classifier(Chain):

    def __init__(self, network, path='../pickle/word_vec.pickle'):
        super(ONT_Classifier, self).__init__(net=network)
        with open(path, 'rb') as f:
            self.word_vec = pickle.load(f)

    def forward(self, xs_a, xs_b, ts):
        loss = self.loss(xs_a, xs_b, ts)
        accu = self.accu(xs_a, xs_b, ts)

        return loss


    def loss(self, xs_a, xs_b, ts):
        ys = self.net(xs_a, xs_b)
        loss = F.softmax_cross_entropy(ys, ts)

        R.report({'loss':loss}, self)
        return loss / len(ts)


    def accu(self, xs_a, xs_b, ts):
        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                ys = self.net(xs_a, xs_b)
        accu = F.accuracy(ys, ts)

        R.report({'accu':accu}, self)
        return accu


    def predict(self, xs_a, xs_b):
        ys = self.net(xs_a, xs_b)

        ys = F.softmax(ys)
        ys = [y.data for y in ys]

        return ys


    def converter(self, batch, device=None):
        xs_a, xs_b, ts = zip(*batch)
        #
        xs_a = [np.stack([self.word_vec[w] for w in x]) for x in xs_a]
        xs_b = [np.stack([self.word_vec[w] for w in x]) for x in xs_b]
        ts = np.array(ts)

        return xs_a, xs_b, ts
