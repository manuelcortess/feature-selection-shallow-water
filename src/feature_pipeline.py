"""
Main pipeline for feature selection and model evaluation in Shallow Water Models.

Note on Cloud Optimization: To process 750,000 spatial-temporal features without 
triggering OOM errors on limited cloud instances, this script utilizes:
1. joblib parallelization for vectorized distance matrix calculations.
2. Strategic subsampling (100k feature limit, 500 simulations) to maintain statistical 
   significance while fitting in memory.
3. CPU-bound execution (GPU disabled) to ensure deterministic memory allocation.
"""

import os
import json
import logging
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Force CPU execution to prevent unmanageable GPU memory spikes on shared cloud instances
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Silencing verbose TensorFlow tracking logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# Setup professional logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
DATA_PATH = "data/optimized_simulations_5k.nc"
RESULTS_DIR = "results"
RANDOM_STATE = 42

# Optimization Constraints
FEATURE_LIMIT = 100000          # Capped to fit memory
SIMULATION_SUBSET = 500         # Subsmapled for distance matrix
TOP_K_FEATURES = 5              # Number of features to extract per parameter

TARGET_PARAMS = ['phic', 'h_rain', 'alpha']
FEATURE_VARS = ['velocity', 'height', 'rain']

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# 1. DATA HANDLING
# ==============================================================================
def load_and_preprocess_data(filepath: str, subset_size: int):
    logging.info(f"Loading dataset from {filepath} (Subset: {subset_size} simulations)...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset missing at {filepath}.")
        
    ds = xr.open_dataset(filepath).isel(simulation=slice(0, subset_size))
    
    features_list = []
    feature_names = []
    for var in FEATURE_VARS:
        var_data = ds[var].values.reshape(subset_size, -1)
        features_list.append(var_data)
        for g in range(ds.dims['grid']):
            for t in range(ds.dims['time']):
                feature_names.append(f"{var}_g{g}_t{t}")
                
    X = np.hstack(features_list)
    
    if X.shape[1] > FEATURE_LIMIT:
        logging.warning(f"Feature space too large ({X.shape[1]}). Truncating to {FEATURE_LIMIT}.")
        X = X[:, :FEATURE_LIMIT]
        feature_names = feature_names[:FEATURE_LIMIT]
        
    targets = {param: ds[param].values for param in TARGET_PARAMS}
    
    logging.info(f"Feature matrix built: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, targets, feature_names


# ==============================================================================
# 2. FEATURE SELECTION METHODS
# ==============================================================================
def compute_information_imbalance(X_col, D_y):
    X_col = X_col.reshape(-1, 1)
    D_x = squareform(pdist(X_col, metric="euclidean"))
    ranks_x = np.argsort(np.argsort(D_x, axis=1), axis=1)
    ranks_y = np.argsort(np.argsort(D_y, axis=1), axis=1)
    imbalance = np.mean(ranks_y[ranks_x == 1]) / (len(X_col) / 2.0)
    return imbalance

def extract_glielmo_features(X, y, feature_names, n_jobs=-1):
    logging.info("Starting Information Imbalance (Glielmo) Selection...")
    D_y = squareform(pdist(y.reshape(-1, 1), metric="euclidean"))
    
    imbalance_scores = Parallel(n_jobs=n_jobs)(
        delayed(compute_information_imbalance)(X[:, i], D_y) 
        for i in range(X.shape[1])
    )
    
    top_indices = np.argsort(imbalance_scores)[:TOP_K_FEATURES]
    top_names = [feature_names[i] for i in top_indices]
    return top_indices, top_names

def extract_nn_backprop_features(X, y, feature_names):
    logging.info("Starting Neural Network Backpropagation Selection...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_scaled, y, epochs=15, batch_size=32, verbose=0)
    
    input_weights = np.abs(model.layers[0].get_weights()[0]).sum(axis=1)
    top_indices = np.argsort(input_weights)[::-1][:TOP_K_FEATURES]
    top_names = [feature_names[i] for i in top_indices]
    return top_indices, top_names


# ==============================================================================
# 3. EVALUATION
# ==============================================================================
def evaluate_model(X, y, model_name: str, apply_log_transform: bool = False):
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    if apply_log_transform:
        y_train = np.log(y_train + 1e-10)
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    preds = rf.predict(X_test_scaled)
    
    if apply_log_transform:
        preds = np.exp(preds) - 1e-10

    results = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
        "time": time.time() - start_time
    }
    
    logging.info(f"[{model_name}] R2: {results['R2']:.4f} | MAE: {results['MAE']:.4f}")
    return results


# ==============================================================================
# 4. VISUALIZATION & PLOTTING 
# ==============================================================================
def plot_model_comparison(results_dict, param_name):
    """
    Recreates the comparative bar charts for MAE, RMSE, and R2.
    """
    models = list(results_dict.keys())
    # Exclude non-model keys from the dictionary
    models = [m for m in models if m not in ['Selected_Feature_Names']]
    
    mae_vals = [results_dict[m]['MAE'] for m in models]
    rmse_vals = [results_dict[m]['RMSE'] for m in models]
    r2_vals = [results_dict[m]['R2'] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.barplot(x=models, y=mae_vals, ax=axes[0], palette="Blues_d")
    axes[0].set_title(f'{param_name} - Mean Absolute Error (Lower is better)')
    
    sns.barplot(x=models, y=rmse_vals, ax=axes[1], palette="Reds_d")
    axes[1].set_title(f'{param_name} - RMSE (Lower is better)')
    
    sns.barplot(x=models, y=r2_vals, ax=axes[2], palette="Greens_d")
    axes[2].set_title(f'{param_name} - R² Score (Higher is better)')
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plots_dir = os.path.join(RESULTS_DIR, 'analysis_plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'performance_comparison_{param_name}.jpg'), dpi=300)
    plt.close()


def compile_final_results():
    """
    Generates the final summary text file combining all parameter results.
    """
    final_results = {}
    for param in TARGET_PARAMS:
        result_file = os.path.join(RESULTS_DIR, f'results_{param}.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                final_results[param] = json.load(f)
    
    with open(os.path.join(RESULTS_DIR, 'final_summary.txt'), 'w') as f:
        f.write("FINAL SUMMARY OF RESULTS\n")
        f.write("="*50 + "\n")
        for param, results in final_results.items():
            f.write(f"\nParameter: {param}\n")
            f.write("-" * 50 + "\n")
            for model, metrics in results.items():
                if model != 'Selected_Feature_Names':
                    f.write(f"{model}:\n")
                    f.write(f"  MAE: {metrics['MAE']:.4f}\n")
                    f.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
                    f.write(f"  R²: {metrics['R2']:.4f}\n")
                    f.write(f"  Time: {metrics.get('time', 0):.2f}s\n")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    logging.info("=== High-Dimensional Feature Selection Pipeline Initialized ===")
    
    X, targets, feat_names = load_and_preprocess_data(DATA_PATH, SIMULATION_SUBSET)
    
    for param in TARGET_PARAMS:
        logging.info(f"\n--- Processing Target Parameter: {param.upper()} ---")
        param_start = time.time()
        y = targets[param]
        needs_log = (param == 'alpha')
        
        idx_glielmo, names_glielmo = extract_glielmo_features(X, y, feat_names)
        idx_nn, names_nn = extract_nn_backprop_features(X, y, feat_names)
        
        results = {
            "All_Features": evaluate_model(X, y, "Full Feature Set", needs_log),
            "Glielmo_Features": evaluate_model(X[:, idx_glielmo], y, "Information Imbalance", needs_log),
            "NN_Features": evaluate_model(X[:, idx_nn], y, "NN Backprop", needs_log),
            "Selected_Feature_Names": {
                "Glielmo": names_glielmo,
                "Neural_Network": names_nn
            }
        }
        
        param_path = os.path.join(RESULTS_DIR, f"results_{param}.json")
        with open(param_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        # Trigger plotting
        plot_model_comparison(results, param)
        
        logging.info(f"Completed {param} in {time.time() - param_start:.2f} seconds")
            
    # Compile the final human-readable summary
    compile_final_results()
        
    logging.info(f"=== Pipeline completed successfully. Data and plots saved to {RESULTS_DIR}/ ===")