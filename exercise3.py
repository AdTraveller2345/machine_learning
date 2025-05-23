from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
housing = fetch_california_housing(as_frame=True)

# The data is in a pandas DataFrame
df = housing.frame

#exercise 1
# Number of samples (rows)
num_samples = df.shape[0]
print("Number of samples in the dataset:", num_samples)

# Number of features (columns), excluding the target
num_features = df.shape[1] - 1
print("Number of features in the dataset, excluding target (price):", num_features)

#exercise 2
# Separate features and target
X = df.drop(columns=["MedHouseVal"])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply polynomial feature expansion (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Report the number of features after expansion
p = X_poly.shape[1]
print("Number of features after polynomial expansion (degree=2):", p)

# exercise 3
# X, X_scaled, X_poly already defined in your code
y = df["MedHouseVal"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_poly_train, X_poly_test, _, _ = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# --- Linear Model ---
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lin = linreg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)

# Get coefficients for MedInc, AveBedrms, HouseAge
feature_names = df.drop(columns=["MedHouseVal"]).columns
coef_dict = dict(zip(feature_names, linreg.coef_))
print("Linear model coefficients:")
print("β_MedInc =", coef_dict["MedInc"])
print("β_AveBedrms =", coef_dict["AveBedrms"])
print("β_HouseAge =", coef_dict["HouseAge"])
print("MSE_eval =", mse_lin)

# --- Polynomial Model ---
polyreg = LinearRegression()
polyreg.fit(X_poly_train, y_train)
y_pred_poly = polyreg.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)

# Get polynomial feature names
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit(X_scaled)
poly_feature_names = poly.get_feature_names_out(feature_names)

poly_coef_dict = dict(zip(poly_feature_names, polyreg.coef_))
print("\nPolynomial model coefficients:")
print("β_MedInc =", poly_coef_dict["MedInc"])
#print("β_MedInc^2 =", poly_coef_dict.get("MedInc^2", "N/A"))
print("β_MedInc AveBedrms =", poly_coef_dict.get("MedInc AveBedrms", "N/A"))
print("β_HouseAge AveBedrms =", poly_coef_dict.get("HouseAge AveBedrms", "N/A"))
print("MSE_eval =", mse_poly)

# Choose a feature to plot against the target
feature_idx = list(feature_names).index("MedInc")
X_plot = X_scaled[:, feature_idx]
X_poly_plot = X_poly[:, :]  # All polynomial features

# For a fair comparison, use the test set
X_test_medinc = X_test[:, feature_idx]
y_test_pred_lin = linreg.predict(X_test)
y_test_pred_poly = polyreg.predict(X_poly_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test_medinc, y_test, color='gray', alpha=0.5, label='Actual data (test set)')
plt.scatter(X_test_medinc, y_test_pred_lin, color='blue', alpha=0.5, label='Linear model predictions')
plt.scatter(X_test_medinc, y_test_pred_poly, color='red', alpha=0.5, label='Polynomial model predictions')
plt.xlabel("Standardized Median Income (MedInc)")
plt.ylabel("Median House Value (MedHouseVal)")
plt.title("Model Fit: MedInc vs MedHouseVal")
plt.legend()
plt.show()

#exercise 4
def beta_ridge(X, y, lam):
    p = X.shape[1]
    return np.linalg.inv(X.T @ X + lam * np.eye(p)) @ X.T @ y

# Set regularization parameter lambda
lam = 0.001

# Compute ridge regression coefficients
ridge_beta = beta_ridge(X_poly_train, y_train, lam)

# Predict on the test set
y_pred_ridge = X_poly_test @ ridge_beta
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

# Print Ridge Polynomial model coefficients without using dict(zip(...))
print("\nRidge Polynomial model coefficients:")
idx_medinc = np.where(poly_feature_names == "MedInc")[0][0]
idx_medinc_avebedrms = np.where(poly_feature_names == "MedInc AveBedrms")[0][0]
idx_houseage_avebedrms = np.where(poly_feature_names == "HouseAge AveBedrms")[0][0]

print("β_MedInc =", ridge_beta[idx_medinc])
print("β_MedInc AveBedrms =", ridge_beta[idx_medinc_avebedrms])
print("β_HouseAge AveBedrms =", ridge_beta[idx_houseage_avebedrms])
print("MSE_eval =", mse_ridge)