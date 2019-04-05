import wikipedia

# 取り込む記事のディレクトリとタイトル
dirPass = '../data/wiki'
#titles = wikipedia.random(pages=10)
titles = ['Machine learning']

# Wikipediaのテキストデータを取り込む
texts = [wikipedia.page(t, auto_suggest=False).content for t in titles]

for (title, text) in zip(titles, texts):
    fName = dirPass + '/' + title.replace(' ', '_').lower() + '.txt'

    with open(fName, 'w') as f:
        f.write(text)
