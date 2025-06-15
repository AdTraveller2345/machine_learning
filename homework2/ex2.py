import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42

def load_iris_data(path=Path("datasets/iris/iris.csv")):
    df = pd.read_csv(path)
    data = df.iloc[:, :-1].values
    labels = df['target'].values
    return data, labels

def print_iris_info(data, labels):
    n_samples, n_features = data.shape
    n_classes = len(np.unique(labels))
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Number of classes: {n_classes}")
    print(f"Feature names: {feature_names}")
    print(f"Class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples")

def init_centroids_greedy_pp(D, r, l=10):
    rng = np.random.default_rng(seed=RANDOM_SEED)
    n, d = D.shape

    # Step 1: Initial random candidates
    candidate_idxs = rng.choice(n, size=l, replace=False)
    sum_dists = [np.sum(np.linalg.norm(D - D[i], axis=1)**2) for i in candidate_idxs]
    best_first_idx = candidate_idxs[np.argmin(sum_dists)]
    X = D[best_first_idx].reshape(1, -1)  # First centroid

    # Step 2: Greedily select remaining centroids
    while X.shape[0] < r:
        # Compute distances to the closest centroid for all points
        dists = np.min([np.linalg.norm(D - x, axis=1)**2 for x in X], axis=0)
        probs = dists / np.sum(dists)
        
        # Sample l candidates with replacement
        candidate_idxs = rng.choice(n, size=l, p=probs, replace=False)
        
        # Select the candidate that would best reduce total distance
        min_total_dist = np.inf
        best_idx = None
        for idx in candidate_idxs:
            new_X = np.vstack([X, D[idx]])
            new_dists = np.min([np.linalg.norm(D - x, axis=1)**2 for x in new_X], axis=0)
            total_dist = np.sum(new_dists)
            if total_dist < min_total_dist:
                min_total_dist = total_dist
                best_idx = idx
        
        X = np.vstack([X, D[best_idx]])

    return X.T  # shape (d, r)

    # Your code here

    indexes = rng.integers(low=0, high=n, size=r)
    X = np.array(D[indexes,:]).T
    return X

# K-means implementation from the lecture slides
def RSS(D,X,Y):
    return np.sum((D- Y@X.T)**2)
    
def getY(labels):
    Y = np.eye(max(labels)+1)[labels]
    return Y
    
def update_centroid(D,Y):
    cluster_sizes = np.diag(Y.T@Y).copy()
    cluster_sizes[cluster_sizes==0]=1
    return D.T@Y/cluster_sizes
    
def update_assignment(D,X):
    dist = np.sum((np.expand_dims(D,2) - X)**2,1)
    labels = np.argmin(dist,1)
    return getY(labels)
    
def kmeans(D,r, X_init, epsilon=0.00001, t_max=10000):
    X = X_init.copy()
    Y = update_assignment(D,X)
    rss_old = RSS(D,X,Y) +2*epsilon
    t=0
    
    #Looping as long as difference of objective function values is larger than epsilon
    while rss_old - RSS(D,X,Y) > epsilon and t < t_max-1:
        rss_old = RSS(D,X,Y)
        X = update_centroid(D,Y)
        Y = update_assignment(D,X)
        t+=1
    print(t,"iterations")
    return X,Y


data, labels = load_iris_data()

data, labels = load_iris_data()
n, d = data.shape
r = 3
l = 10

# Initialize centroids
X_init = init_centroids_greedy_pp(data, r, l)

# Run k-means
X_final, Y_final = kmeans(data, r, X_init)

# Recover cluster labels from one-hot Y
cluster_labels = np.argmax(Y_final, axis=1)

from sklearn.metrics import normalized_mutual_info_score

# Mean approximation error
approx_error = RSS(data, X_final, Y_final) / (n * d)
print(f"Mean approximation error: {approx_error:.4f}")

# Normalized Mutual Information
nmi = normalized_mutual_info_score(labels, cluster_labels)
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

