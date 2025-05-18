from sklearn.datasets import fetch_20newsgroups

categories = ['sci.space', 'misc.forsale', 'comp.graphics', 'rec.sport.hockey']
train = fetch_20newsgroups(subset='train', categories=categories)
test  = fetch_20newsgroups(subset='test',  categories=categories)

# My implementation

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# remove headers/footers/quotes for a more realistic topic classifier:
train = fetch_20newsgroups(subset='train',
                           categories=categories,
                           remove=('headers','footers','quotes'))
test = fetch_20newsgroups(subset='test',
                          categories=categories,
                          remove=('headers','footers','quotes'))

# build binary bag‑of‑words
vectorizer = CountVectorizer(binary=True)
X_train = vectorizer.fit_transform(train.data)
y_train = train.target

# vocabulary
vocab = vectorizer.get_feature_names_out()

print(y_train)
print(train.target_names)

# Ask about the change in category order!!!!!!
