from pathlib import Path
import numpy as np
import pandas as pd


def load_ratings_data_pandas(data_dir="ml-latest-small/"):
    """Load data using pandas dataframes."""
    data_dir = Path(data_dir)
    assert data_dir.exists(), f"{data_dir} does not exist"

    return pd.read_csv(data_dir / 'ratings.csv',sep=',')


def load_movies_data_pandas(data_dir="ml-latest-small/"):
    """Load data using pandas dataframes."""
    data_dir = Path(data_dir)
    assert data_dir.exists(), f"{data_dir} does not exist"
    return pd.read_csv(data_dir / 'movies.csv')


def filter_data(ratings_data: pd.DataFrame, movies_data: pd.DataFrame):
    """Filter data. Too little ratings prevent effective use of matrix completion."""
    ratings_data = ratings_data.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

    keep_movie = (ratings_data != 0).sum(axis=0) > 100
    ratings_data = ratings_data.loc[:, keep_movie]

    # Filter movies_data by movieId (columns of ratings_data after filtering)
    movies_data = movies_data[movies_data['movieId'].isin(ratings_data.columns)]

    keep_user = (ratings_data != 0).sum(axis=1) >= 5
    ratings_data = ratings_data.loc[keep_user, :]

    return ratings_data, movies_data


def print_data_summary(ratings: pd.DataFrame):
    n_users = ratings.shape[0]
    n_movies = ratings.shape[1]
    n_ratings = (ratings != 0).sum().sum()
    density = n_ratings / (n_users * n_movies)

    print(f"Dataset Summary")
    print(f"----------------")
    print(f"Users: {n_users}")
    print(f"Movies: {n_movies}")
    print(f"Total Ratings: {n_ratings}")
    print(f"Data Density: {density:.4f} (fraction of observed ratings)")


def load_ratings_data(data_dir="ml-latest-small/", print_summary=False):
    """Load data in numpy format."""
    ratings, movies = filter_data(
        load_ratings_data_pandas(data_dir=data_dir),
        load_movies_data_pandas(data_dir=data_dir)
    )
    if print_summary:
        print_data_summary(ratings)
    return ratings.to_numpy()


def matrix_completion(D, n_features, n_movies, n_users, t_max=100, lambd=0.1):
    np.random.seed(0)
    X = np.random.normal(size=(n_movies, n_features))
    Y = np.random.normal(size=(n_users, n_features))

    # Implementation the optimization procedure here
    O = (D != 0).astype(float)

    for t in range(t_max):
        for i in range(n_users):
            idx = O[i, :] == 1
            X_i = X[idx]
            D_i = D[i, idx]
            if len(D_i) == 0:
                continue
            A = X_i.T @ X_i + lambd * np.eye(n_features)
            b = X_i.T @ D_i
            Y[i] = np.linalg.solve(A, b)

        for j in range(n_movies):
            idx = O[:, j] == 1
            Y_j = Y[idx]
            D_j = D[idx, j]
            if len(D_j) == 0:
                continue
            A = Y_j.T @ Y_j + lambd * np.eye(n_features)
            b = Y_j.T @ D_j
            X[j] = np.linalg.solve(A, b)

    return X, Y          

def compute_error(D, X, Y):
    O = (D != 0).astype(float)
    pred = Y @ X.T
    error = np.sum(((O * (D - pred)) ** 2)) / np.sum(O)
    return error

ratings = load_ratings_data("datasets/ml-latest-small", print_summary=True)
n_users, n_movies = ratings.shape

iterations = 100
lambd = 0.1 # change here for b and c

X, Y = matrix_completion(ratings, 20, n_movies, n_users, iterations, lambd)
error = compute_error(ratings, X, Y)

print(f"Average squared error after 100 iterations: {error:.4f}")

ratings_matrix = Y @ X.T  
first_user_ratings = ratings_matrix[0] 
movie_names = ["Jumanji (1995)", "Fight Club (1999)", "Matrix, The (1999)", "Monty Python and the Holy Grail (1975)"]
_, movies_data = filter_data(
    load_ratings_data_pandas("datasets/ml-latest-small"),
    load_movies_data_pandas("datasets/ml-latest-small")
)

for name in movie_names:
    row = movies_data[movies_data['title'] == name]
    col_idx = movies_data.index.get_loc(row.index[0])
    est_rating = first_user_ratings[col_idx]
    print(f"Estimated rating for '{name}' by first user: {est_rating:.2f}")
