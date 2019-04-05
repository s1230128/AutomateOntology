import numpy as np
import pickle
import random
import nltk


# データの読み込み
with open('../pickle/wordnet_lemmas.pickle', 'rb') as f:
    lemmas = pickle.load(f)

with open('../pickle/wordnet_lemma_pears.pickle', 'rb') as f:
    synonyms, supsubs, subsups = pickle.load(f)

with open('../pickle/word_vec.pickle', 'rb') as f:
    word_vec = pickle.load(f)

# Facebook word_vec に登録されていない単語を持つものを除外
lemmas = [l.split('_') for l in lemmas]
lemmas = [l for l in lemmas if all([w in word_vec for w in l])]

synonyms = [(a.split('_'), b.split('_')) for a, b in synonyms]
synonyms = [(a, b) for a, b in synonyms        \
            if  all([w in word_vec for w in a])\
            and all([w in word_vec for w in b])]
supsubs  = [(a.split('_'), b.split('_')) for a, b in supsubs]
supsubs  = [(a, b) for a, b in supsubs         \
            if  all([w in word_vec for w in a])\
            and all([w in word_vec for w in b])]
subsups  = [(a.split('_'), b.split('_')) for a, b in subsups]
subsups  = [(a, b) for a, b in subsups         \
            if  all([w in word_vec for w in a])\
            and all([w in word_vec for w in b])]

# 関係性のないペアのランダム生成
lemmas = list(lemmas)

related = set()
for a, b in synonyms + supsubs + subsups:
    key = ' '.join(('_'.join(a), '_'.join(b)))
    related.add(key)

unrelated = []
while len(unrelated) < 500000:
    a = random.choice(lemmas)
    b = random.choice(lemmas)

    key = ' '.join(('_'.join(a), '_'.join(b)))
    if key in related: continue

    unrelated.append((a, b))

#
unrelated = [(a, b, 0) for a, b in unrelated]
synonyms  = [(a, b, 1) for a, b in synonyms]
supsubs   = [(a, b, 2) for a, b in supsubs]
subsups   = [(a, b, 3) for a, b in subsups]

data = unrelated + synonyms + supsubs + subsups
random.shuffle(data)

print('lemmas    :', len(lemmas))
print('synonyms  :', len(synonyms))
print('supsubs   :', len(supsubs))
print('subsups   :', len(subsups))
print('unrelated :', len(unrelated))
print('total     :', len(data))

# 保存
with open('../pickle/data.pickle', 'wb') as f: pickle.dump(data , f)

print('finish!')
