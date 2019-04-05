import numpy as np
import pickle
from gensim.models import KeyedVectors


# データの読み込み
with open('../pickle/hypo_pears.pickle', 'rb') as f:
    hypos = pickle.load(f)

#
word2vec = KeyedVectors.load_word2vec_format('../../research/data/wordVector/wiki.en.vec')
print('loading finish!')

#
word_set = set()
for a, b in hypos:
    for w in a: word_set.add(w)
    for w in b: word_set.add(w)

#
word_list = list(word_set)

#
word_vec = dict()
for w in word_list:
    if w in word2vec:
        word_vec[w] = word2vec[w]

#
with open('../pickle/word_vec.pickle', 'wb') as f:
    pickle.dump(word_vec, f)

print('finish!')
