from pathlib import Path
import json
import pandas as pd
import numpy as np

def load_kdd_data(data_dir="./datasets/kdd_balanced"):
    """
    Load the balanced dataset from Parquet format (ultra-fast loading).
    
    Returns:
        D_balanced: Feature matrix (numpy array)
        is_normal_balanced: Boolean labels (numpy array)  
        original_indices: Original indices from full dataset (numpy array)
        df_balanced: Original dataframe subset (if available)
        metadata: Dataset metadata
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import json
    
    data_path = Path(data_dir)
    
    # Check if required files exist
    required_files = ["balanced_dataset.parquet", "metadata.json"]
    missing_files = [f for f in required_files if not (data_path / f).exists()]
    if missing_files:
        raise FileNotFoundError(f"Required files not found in {data_path}: {missing_files}")
    
    # Load main dataset (this is very fast with Parquet)
    df_main = pd.read_parquet(data_path / "balanced_dataset.parquet")
    
    # Extract components
    feature_cols = [col for col in df_main.columns if col.startswith('feature_')]
    D_balanced = df_main[feature_cols].values
    is_normal_balanced = df_main['is_normal'].values
    original_indices = df_main['original_index'].values
    
    # Load original dataframe if available
    df_balanced = None
    if (data_path / "original_data_balanced.parquet").exists():
        df_balanced = pd.read_parquet(data_path / "original_data_balanced.parquet")
    
    # Load metadata
    with open(data_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"Balanced dataset loaded from {data_path} (Parquet format)")
    print(f"  Features: {D_balanced.shape}")
    print(f"  Normal: {metadata['n_normal']:,}, Intrusion: {metadata['n_intrusion']:,}")
    
    return D_balanced, is_normal_balanced, original_indices, df_balanced, metadata

D_balanced, is_normal_balanced, original_indices, _, metadata = load_kdd_data()

from sklearn.decomposition import TruncatedSVD
import numpy as np

# 1a
mu_C = np.mean(D_balanced, axis=0)
C = D_balanced - mu_C 

svd = TruncatedSVD(n_components=10)
Z = svd.fit_transform(C) 

z0, z1, z2 = Z[7, 0], Z[7, 1], Z[7, 2]

print("z0 =", z0)
print("z1 =", z1)
print("z2 =", z2, "\n")

# 1b
X = svd.components_
D_hat = Z @ X + mu_C
errors = np.sum((D_balanced - D_hat)**2, axis=1)

error_8th = errors[7]

mean_intrusion_error = np.mean(errors[~is_normal_balanced])

mean_normal_error = np.mean(errors[is_normal_balanced])

print("Reconstruction error of 8th data point:", error_8th)
print("Mean reconstruction error (intrusions):", mean_intrusion_error)
print("Mean reconstruction error (normals):", mean_normal_error, "\n")

# 1c
sorted_errors = errors[np.argsort(errors)]
threshold_90 = np.percentile(errors, 90)
threshold_95 = np.percentile(errors, 95)
threshold_99 = np.percentile(errors, 99)
outliers_90 = errors >= threshold_90
outliers_95 = errors >= threshold_95
outliers_99 = errors >= threshold_99

tp_90 = np.sum(outliers_90 & ~is_normal_balanced)
fp_90 = np.sum(outliers_90 & is_normal_balanced)

tp_95 = np.sum(outliers_95 & ~is_normal_balanced)
fp_95 = np.sum(outliers_95 & is_normal_balanced)

tp_99 = np.sum(outliers_99 & ~is_normal_balanced)
fp_99 = np.sum(outliers_99 & is_normal_balanced)

print("TP at 90% threshold:", tp_90)
print("FP at 90% threshold:", fp_90)
print("TP at 95% threshold:", tp_95)
print("FP at 95% threshold:", fp_95)
print("TP at 99% threshold:", tp_99)
print("FP at 99% threshold:", fp_99)