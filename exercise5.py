from sklearn.datasets import fetch_20newsgroups

categories = ['sci.space', 'misc.forsale', 'comp.graphics', 'rec.sport.hockey']
train = fetch_20newsgroups(subset='train', categories=categories)
test  = fetch_20newsgroups(subset='test',  categories=categories)

# print(train.DESCR)
# My implementation

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


# Map class names to indices correctly since sklearn takes categories alphabetically
print(train.target_names)
print(categories)
print(train.target)
print(np.bincount(train.target))
mapping = { orig_idx: categories.index(name)
            for orig_idx,name in enumerate(train.target_names) }

y = np.array([mapping[orig] for orig in train.target])
# (Optional) override train.target and train.target_names if you want
# train.target      = y
# train.target_names = categories
print(y)
print(np.bincount(y))

# 5a
print("5a:")
n = len(y)
priors = []
for cls in range(4):
    prior = np.sum(y == cls) / n
    priors.append(prior)
    print(f"p(y={cls}) = {prior:.2f}")

# 5b
print("5b:")
vec = CountVectorizer(stop_words="english", min_df=5,token_pattern="[^\W\d_]+", binary=True)
X = vec.fit_transform(train.data)
idx_chip = vec.vocabulary_.get("chip")
alpha = 1e-5
vocab = vec.get_feature_names_out()
# print(idx_chip)
print(vocab[idx_chip])

# for each class, count and compute log‐prob
for cls in range(4):
    mask = (y == cls)
    N_cls = np.sum(mask)
    N_chip = np.sum(X[mask, idx_chip])
    # |X_k| is 2 since binary
    p = (N_chip + alpha) / (N_cls + 2*alpha)
    print(f"log p(chip=1 | y={cls}) = {np.log(p):.2f}")

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
    print(f"p(y={cls} | x['{word}']=1) = {post:.2f}")

# Check
print(f"{conditional("electronics", 0) * priors[0] / np.mean(X[:, vec.vocabulary_.get("electronics")]):.2f}")
print(f"{conditional("sale", 1) * priors[1] / np.mean(X[:, vec.vocabulary_.get("sale")]):.2f}")
print(f"{conditional("games", 2) * priors[2] / np.mean(X[:, vec.vocabulary_.get("games")]):.2f}")
print(f"{conditional("ball", 3) * priors[3] / np.mean(X[:, vec.vocabulary_.get("ball")]):.2f}")





