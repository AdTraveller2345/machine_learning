from sklearn.datasets import fetch_20newsgroups

categories = ['sci.space', 'misc.forsale', 'comp.graphics', 'rec.sport.hockey']
train = fetch_20newsgroups(subset='train', categories=categories)
test  = fetch_20newsgroups(subset='test',  categories=categories)

# print(train.DESCR)
# My implementation

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# remove headers/footers/quotes for a more realistic classifier
# train = fetch_20newsgroups(subset='train',
#                            categories=categories,
#                            remove=('headers','footers','quotes'))
# test = fetch_20newsgroups(subset='test',
#                           categories=categories,
#                           remove=('headers','footers','quotes'))

# Map class names to indices correctly since sklearn takes categories alphabetically
print(train.target)
mapping = { orig_idx: categories.index(name)
            for orig_idx,name in enumerate(train.target_names) }

y = np.array([mapping[orig] for orig in train.target])
# (Optional) override train.target and train.target_names if you want
# train.target      = y
# train.target_names = categories
print(y)

# 5a
print("5a:")
n = len(y)
priors = []
for cls in range(4):
    prior = np.sum(y == cls) / n
    priors.append(prior)
    print(f"p(y={cls}) = {prior:.4f}")

# 5b
print("5b:")
vec = CountVectorizer(binary=True)
X = vec.fit_transform(train.data)
idx_chip = vec.vocabulary_.get("chip")
alpha = 1e-5
vocab = vec.get_feature_names_out()
print(vocab[idx_chip])

# for each class, count and compute log‐prob
for cls in range(4):
    mask = (y == cls)
    N_cls = np.sum(mask)
    N_chip = np.sum(X[mask, idx_chip])
    # |X_k| is 2 since binary
    p = (N_chip + alpha) / (N_cls + 2*alpha)
    print(f"log p(chip=1 | y={cls}) = {np.log(p):.4f}")

# 5c
print("5c:")
# print(priors)

# Helper to compute p(x_word=1 | y=cls) with Laplace smoothing
def conditional(word, cls, alpha=1e-5):
    idx = vec.vocabulary_.get(word)
    mask = (y == cls)
    N_cls = np.sum(mask)
    N_word = np.sum(X[mask, idx])
    return (N_word + alpha) / (N_cls + 2*alpha)

# Words per class
words = ["electronics", "sale", "games", "ball"]

# Compute posteriors
posteriors = []
for cls, word in enumerate(words):
    p_x_given_y = conditional(word, cls)
    # denominator: sum over all classes
    denom = sum(conditional(word, k) * priors[k] for k in range(4))
    posterior = (p_x_given_y * priors[cls]) / denom
    posteriors.append(posterior)

# Display
for cls, word, post in zip(range(4), words, posteriors):
    print(f"p(y={cls} | x['{word}']=1) = {post:.4f}")

# Check
# print(conditional("ball", 3) * priors[3] / np.mean(X[:, vec.vocabulary_.get("ball")]))




