import pickle
import numpy as np
import chainer.functions as F
import chainer.iterators as I
import chainer.optimizers as O
import chainer.training  as T
import chainer.serializers as S
import chainer.training.extensions as E
from ONT import *
from ONT_Classifier import *


with open('../pickle/concepts.pickle', 'rb') as f:
    ngram_concepts = pickle.load(f)

with open('../pickle/word_vec.pickle', 'rb') as f:
    word_vec = pickle.load(f)

# {ngram:conceptの数}
ns_concept = {1:20, 2:20, 3:20}
concepts = []
for k, v in ns_concept.items():
    tfidf = list(ngram_concepts[k].items())
    tfidf.sort(reverse=True, key=lambda t: t[1])
    concepts.extend(tfidf[:v])

concepts.sort(reverse=True, key=lambda t: t[1])
print(' ------------------------------------------------- ')
print('|  tfidf    | n-gram                              |')
print('|-----------|-------------------------------------|')
for c, v in concepts: print('|  {:>7.3f}  |  {:<33}  |'.format(v, c))


# concepts = ['computer', 'device', 'digital computer', 'machine language']

# concepts = [c.split() for c in concepts]
# concepts = [c for c in concepts if all(w in word_vec for w in c)]
# concepts_vec = [np.stack([word_vec[w] for w in c]) for c in concepts]
# concepts = ['_'.join(c) for c in concepts]
#
# pears = []
# for c_a in concepts:
#     for c_b in concepts:
#         pears.append((c_a, c_b))
#
# xs_a = []
# xs_b = []
# for v_a in concepts_vec:
#     for v_b in concepts_vec:
#         xs_a.append(v_a)
#         xs_b.append(v_b)

# n_rnn = 1
# size_hidden = 300
# rate_dropout = 0.1
#
# net = ONT_BiGRU(n_rnn, size_hidden, rate_dropout)
# model = ONT_Classifier(net)
# S.load_npz("../model_epoch-20", model)
#
# ts = model.predict(xs_a, xs_b)

# t0 : unrelated
# t1 : synonym
# t2 : sup-sub
# t3 : sub-sup
# print('synonym : sup-sub : sub-sup : unrelated :')
# for (a, b), (t0, t1, t2, t3) in zip(pears, ts):
    # print('{:>16} - {:<16} : {:>6.3f} : {:6.3f} : {:>6.3f} :  {:>6.3f}'.format(a, b, t1, t2, t3, t0))
    # print('{:>16} {:<16} {:>6.3f} {:6.3f} {:>6.3f} {:>6.3f}'.format(a, b, t1, t2, t3, t0))

# results = list(zip(pears, ts))
#
# results.sort(reverse=True, key=lambda t: t[1][1])
# print('\n==== synonym ====')
# for (a, b), (_, t, _, _) in results[:50]:
#     print('{:>6.3f} : {:>30} - {:<30}'.format(t, a, b))
#
# results.sort(reverse=True, key=lambda t: t[1][2])
# print('\n==== sup-sub ====')
# for (a, b), (_, _, t, _) in results[:50]:
#     print('{:>6.3f} : {:>30} - {:<30}'.format(t, a, b))
#
# results.sort(reverse=True, key=lambda t: t[1][3])
# print('\n==== sub-sup ====')
# for (a, b), (_, _, _, t) in results[:50]:
#     print('{:>6.3f} : {:>30} - {:<30}'.format(t, a, b))
