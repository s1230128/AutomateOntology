from nltk.corpus import wordnet as wn
from pprint import pprint
import pickle

# VERB, NOUN, ADJ, ADV
# 'v' , 'n' ,
# wn.synsets() : 同義語のリスト
# hypernyms : list(synset) : 上位語
# hyponyms  : list(synset) : 下位語
synsets = list(wn.all_synsets(pos='n'))


# wordnetに登録されているlemma一覧を作成
lemmas = set()
for s in synsets:
    for l in s.lemma_names():
        lemmas.add(l)

# 同義語のペア作成
synonyms = []
for s in synsets:
    for l_a in s.lemma_names():
        for l_b in s.lemma_names():
            synonyms.append((l_a, l_b))


# 上位下位、下位上位のペア作成
supsubs = []
subsups = []
for s in synsets:
    hypos = s.hyponyms()
    for h in hypos:
        for l_a in s.lemma_names():
            for l_b in h.lemma_names():
                supsubs.append((l_a, l_b))
                subsups.append((l_b, l_a))

print('lemmas   :', len(lemmas))
print('synonyms :', len(synonyms))
print('supsubs  :', len(supsubs))
print('subsups  :', len(subsups))

with open('../pickle/wordnet_lemmas.pickle', 'wb') as f:
    pickle.dump(lemmas, f)

with open('../pickle/wordnet_lemma_pears.pickle', 'wb') as f:
    pickle.dump((synonyms, supsubs, subsups), f)
