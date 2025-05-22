from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print(df)
# print(iris.DESCR)

# My implementation

def entropy(labels):
    """Calculate entropy using natural logarithm."""
    counts = np.bincount(labels)
    probabilities = counts[counts > 0] / len(labels)
    return -np.sum(probabilities * np.log(probabilities))

# 6a: Root node entropy
E_root = entropy(df['target'])
print("6a, Root node entropy:", E_root)
# print(np.bincount(df['target']))

# 6b: Split at mean sepal width
split_value = np.mean(df['sepal width (cm)'])
# print("Split value:", split_value)
left = df[df['sepal width (cm)'] <= split_value]
# print(left)
right = df[df['sepal width (cm)'] > split_value]
# print(right)

E_left = entropy(left['target'])
E_right = entropy(right['target'])
# print(E_left, E_right)

# Weighted average entropy after split
weighted_entropy = (len(left) / len(df)) * E_left + (len(right) / len(df)) * E_right

# Information gain
IG = E_root - weighted_entropy
print("6b, IG:", IG)
print("---------------------------------")
print(f"6a, Root node entropy: {E_root:.2f}")
print(f"6b, IG: {IG:.2f}")