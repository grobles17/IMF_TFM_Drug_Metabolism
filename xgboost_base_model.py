"""
Multi-CYP classification using binary relevance (one XGBoost per CYP).
Global hyperparameter search over all CYPs, evaluated by macro MCC on 3-fold CV.
Final evaluation on an untouched 20% test set.
"""

import os
import json
import warnings
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, f1_score, hamming_loss
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import joblib

import ast

# ------------------------------  CONFIGURATION  ---------------------------------
RANDOM_SEED = 33
TEST_FRACTION = 0.2
CV_FOLDS = 3
OUTPUT_DIR = "./xgboost_models"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameter grid
param_grid = {
    "learning_rate": [0.05, 0.1, 0.3],
    "max_depth": [3, 6, 9],
    "n_estimators": [100, 200],
}
# Fixed XGBoost parameters (basic setup)
XGB_FIXED = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "verbosity": 0,
    "n_jobs": 1,          # single thread for reproducibility
}

# List of all CYP classes
cyp_labels = [  
    "CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "CYP3A5", 
    "CYP2C19", "CYP2C8", "CYP2B6", "CYP3A7", "CYP2E1", 
    "CYP1A1", "CYP2A6", "CYP3A43"
]
# -------------------------------------------------------------------------------


def load_data(fp_path, labels_path, cyp_list):
    """
    Load representation and CYP labels.

    Args:
        fp_path (str): TSV file with ID in first column, then representation values.
        labels_path (str): CSV file with columns 'DrugBank ID' and 'CYPs'
        cyp_list (list): All possible CYP names.

    Returns:
        X (pd.DataFrame): Representation, index = molecule ID.
        Y (pd.DataFrame): Binary matrix (molecules x CYPs), index = same ID.
    """
    repr_df = pd.read_csv(fp_path, sep="\t", index_col=0)
    print(f"Loaded representation: {repr_df.shape}")

    labels_df = pd.read_csv(labels_path, sep=",")
    all_cyps = cyp_list
    Y = pd.DataFrame(0, index=labels_df["DrugBank ID"], columns=all_cyps)

    for _, row in labels_df.iterrows():
        mol_id = row["DrugBank ID"]
        try:
            cyps = ast.literal_eval(str(row["CYPs"]))  # parse the list string properly
        except (ValueError, SyntaxError):
            cyps = []
            print(f"Warning: Could not parse CYPs for molecule {mol_id}. Setting to empty list.")
        for c in cyps:  # now iterating over actual CYP names
            if c in Y.columns:
                Y.loc[mol_id, c] = 1

    common_ids = repr_df.index.intersection(Y.index)
    X = repr_df.loc[common_ids]
    Y = Y.loc[common_ids]
    print(f"Common molecules: {len(common_ids)}")
    return X, Y

def iterative_stratified_split(X, Y, test_size=0.2, random_state=33):
    """
    Multilabel-preserving train/test split.
    Returns (X_train, X_test, Y_train, Y_test) as DataFrames.
    """
    X_arr = X.values
    Y_arr = Y.values

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(msss.split(X_arr, Y_arr))

    X_train = pd.DataFrame(X_arr[train_idx], index=X.index[train_idx], columns=X.columns)
    X_test  = pd.DataFrame(X_arr[test_idx],  index=X.index[test_idx],  columns=X.columns)
    Y_train = pd.DataFrame(Y_arr[train_idx], index=X.index[train_idx], columns=Y.columns)
    Y_test  = pd.DataFrame(Y_arr[test_idx],  index=X.index[test_idx],  columns=Y.columns)

    return X_train, X_test, Y_train, Y_test

def check_cyp_distribution(Y_train, Y_test, cyp_list, min_positives=5):
    """
    Warn if any CYP has too few positives in train or test.
    Returns a list of CYPs that are safe to evaluate (>= min_positives in both).
    """
    safe_cyps = []
    for cyp in cyp_list:
        pos_train = Y_train[cyp].sum()
        pos_test = Y_test[cyp].sum()
        if pos_train < min_positives or pos_test < min_positives:
            warnings.warn(
                f"CYP {cyp}: train positives={pos_train}, test positives={pos_test}. "
                f"MCC may be unstable. Skipping from macro MCC."
            )
        else:
            safe_cyps.append(cyp)
    return safe_cyps


def train_and_evaluate_cyp(X_train, y_train, X_val, y_val, params):
    """
    Train a single XGBoost model on (X_train, y_train) and evaluate MCC on val.
    Returns: MCC (float)
    """
    # Merge fixed and tuned parameters
    model_params = {**XGB_FIXED, **params}
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    try:
        mcc = matthews_corrcoef(y_val, y_pred)
    except Exception:
        mcc = 0.0
    return mcc


def cross_validate_cyp(X, y, fold_indices, params):
    """
    Perform 3-fold CV for one CYP using pre-computed fold indices.
    fold_indices: list of tuples (train_idx, val_idx) with indices into X.
    Returns: average MCC across folds.
    """
    mccs = []
    for train_idx, val_idx in fold_indices:
        X_train_f = X.iloc[train_idx]
        y_train_f = y.iloc[train_idx]
        X_val_f = X.iloc[val_idx]
        y_val_f = y.iloc[val_idx]
        mcc = train_and_evaluate_cyp(X_train_f, y_train_f, X_val_f, y_val_f, params)
        mccs.append(mcc)
    return np.mean(mccs)


def grid_search_over_cyps(X_train, Y_train, cyp_list, fold_indices, param_grid):
    """
    Exhaustive grid search over hyperparameters.
    For each combination: compute macro MCC (mean over CYPs of CV MCC).
    Returns: results DataFrame, best_params dict.
    """
    # Unroll parameter grid
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))
    results = []

    for combo_idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"\n[{combo_idx}/{len(combos)}] Testing params: {params}")

        per_cyp_mcc = []
        for cyp in cyp_list:
            y_cyp = Y_train[cyp]
            # Skip if all zeros (should not happen by design)
            if y_cyp.sum() == 0:
                per_cyp_mcc.append(0.0)
                continue
            mcc_avg = cross_validate_cyp(X_train, y_cyp, fold_indices, params)
            per_cyp_mcc.append(mcc_avg)
            print(f"  {cyp}: avg MCC = {mcc_avg:.4f}")

        macro_mcc = np.mean(per_cyp_mcc) if per_cyp_mcc else 0.0
        results.append({
            "params": params,
            "macro_mcc": macro_mcc,
            "per_cyp_mcc": per_cyp_mcc,
            "cyp_order": cyp_list  # to map later
        })
        print(f"  -> Macro MCC = {macro_mcc:.4f}")

    # Find best macro MCC, tie‑break by variance of per‑CYP MCCs
    best_macro = max(r["macro_mcc"] for r in results)
    candidates = [r for r in results if abs(r["macro_mcc"] - best_macro) < 1e-6]
    if len(candidates) > 1:
        # Compute variance of per‑CYP MCCs (across CYPs) for each candidate
        for cand in candidates:
            cand["variance"] = np.var(cand["per_cyp_mcc"])
        best_candidate = min(candidates, key=lambda x: (x["variance"], candidates.index(x)))
    else:
        best_candidate = candidates[0]

    best_params = best_candidate["params"]
    print(f"\nBest parameters: {best_params} (macro MCC = {best_macro:.4f})")

    # Convert results to DataFrame for saving
    results_df = pd.DataFrame([
        {"param_set": idx, "params": str(r["params"]), "macro_mcc": r["macro_mcc"]}
        for idx, r in enumerate(results)
    ])
    return results_df, best_params


def retrain_all_cyps(X_train, Y_train, cyp_list, best_params, models_dir):
    """
    Train final model for each CYP on full training set.
    Saves each model as .joblib.
    Returns: dict of trained models.
    """
    models = {}
    for cyp in cyp_list:
        print(f"Retraining {cyp} on full 80% set...")
        y_cyp = Y_train[cyp]
        if y_cyp.sum() == 0:
            warnings.warn(f"CYP {cyp} has no positives in training - skipping model.")
            continue
        model_params = {**XGB_FIXED, **best_params}
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_cyp)
        models[cyp] = model
        joblib.dump(model, os.path.join(models_dir, f"{cyp}.joblib"))
    return models


def evaluate_test_set(models, X_test, Y_test, cyp_list):
    """
    Evaluate all models on the unseen test set.
    Returns: per-CYP MCC, macro MCC, micro F1, hamming loss.
    """
    y_true_list = []
    y_pred_list = []
    per_cyp_mcc = {}

    for cyp in cyp_list:
        if cyp not in models:
            per_cyp_mcc[cyp] = 0.0
            continue
        model = models[cyp]
        y_true = Y_test[cyp].values
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= 0.5).astype(int)
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except:
            mcc = 0.0
        per_cyp_mcc[cyp] = mcc

    # Build full multilabel prediction matrix
    Y_pred_matrix = np.column_stack(y_pred_list)   # samples x CYPs
    Y_true_matrix = Y_test[cyp_list].values
    micro_f1 = f1_score(Y_true_matrix, Y_pred_matrix, average="micro")
    ham_loss = hamming_loss(Y_true_matrix, Y_pred_matrix)

    macro_mcc = np.mean(list(per_cyp_mcc.values()))
    return per_cyp_mcc, macro_mcc, micro_f1, ham_loss


def main():
    print("=" * 60)
    print("Multi-CYP XGBoost training pipeline")
    print("=" * 60)

    # 1. Load data
    # --- MODIFY PATHS AS NEEDED ---
    fp_file = "morgan_output_representation.tsv"
    labels_file = "DrugBank_curated_df.csv"  
    # ------------------------------
    X, Y = load_data(fp_file, labels_file, cyp_labels)
    print(f"Total molecules: {X.shape[0]}, CYP classes: {Y.shape[1]}")

    # 2. Iterative stratified split (80/20)
    X_train, X_test, Y_train, Y_test = iterative_stratified_split(
        X, Y, test_size=TEST_FRACTION, random_state=RANDOM_SEED
    )
    print(f"Training set: {X_train.shape[0]} molecules")
    print(f"Test set:     {X_test.shape[0]} molecules")

    # 3. Check CYP distributions and keep only CYPs with sufficient positives
    safe_cyps = check_cyp_distribution(Y_train, Y_test, cyp_labels, min_positives=5)
    print(f"CYPs with sufficient data in both train/test: {safe_cyps}")
    if len(safe_cyps) == 0:
        raise RuntimeError("No CYP has enough positives - cannot continue.")
    # Use only safe CYPs for macro MCC calculation
    used_cyps = safe_cyps

    # 4. Create fixed 3‑fold indices on the training set
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_indices = list(kf.split(X_train))  # each element: (train_idx, val_idx)
    print(f"Created {CV_FOLDS} CV folds on training data (fixed seed).")

    # 5. Grid search over all hyperparameters
    results_df, best_params = grid_search_over_cyps(
        X_train, Y_train, used_cyps, fold_indices, param_grid
    )

    # Save CV results
    results_df.to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # 6. Retrain on full 80% with best parameters
    models = retrain_all_cyps(X_train, Y_train, used_cyps, best_params, MODELS_DIR)

    # 7. Final evaluation on untouched 20% test set
    per_cyp_mcc, macro_mcc, micro_f1, ham_loss = evaluate_test_set(
        models, X_test, Y_test, used_cyps
    )

    # Save test metrics
    test_metrics = {
        "macro_mcc": macro_mcc,
        "micro_f1": micro_f1,
        "hamming_loss": ham_loss,
        "per_cyp_mcc": per_cyp_mcc,
    }
    with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS (20% unseen data)")
    print(f"Macro MCC      : {macro_mcc:.4f}")
    print(f"Micro F1       : {micro_f1:.4f}")
    print(f"Hamming loss   : {ham_loss:.4f}")
    print("\nPer-CYP MCC:")
    for cyp, mcc in per_cyp_mcc.items():
        print(f"  {cyp}: {mcc:.4f}")
    print("=" * 60)
    print(f"All models and results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()