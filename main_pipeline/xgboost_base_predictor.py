"""
Multi-CYP classification using binary relevance (one XGBoost per CYP).
Global hyperparameter search (learning_rate, max_depth, n_estimators).
For each CYP and each hyperparameter set, threshold is optimised during CV
(by maximising average validation MCC over 0.1, 0.3, 0.5, 0.7).
Final evaluation on untouched 20% test set uses per-CYP optimal thresholds.
"""

import os
import json
import warnings
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, f1_score, hamming_loss
import joblib
import ast

# CONFIGURATION
RANDOM_SEED = 33

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
LABELS_FILE = os.path.normpath(
    os.path.join(REPO_ROOT, "DataBases", "DrugBank_curated_df.csv")
    )
SPLITS_PATH = os.path.join(SCRIPT_DIR, "splits", "benchmark_splits.joblib")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models", "xgboost_models")

# Hyperparameter grid
param_grid = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [6, 9],
    "n_estimators": [100, 200, 300],
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

# Thresholds to evaluate during CV (per CYP)
THRESHOLDS_TO_TRY = [0.1, 0.3, 0.5, 0.7]

def load_data(fp_path: str, labels_path: str, cyp_list: list) -> tuple:
    """
    Load fingerprint representations and binary CYP labels.

    Parameters
    ----------
    fp_path : str
        Path to a tab-separated file where the first column is the molecule ID
        and the remaining columns are feature values.
    labels_path : str
        Path to a CSV file with columns 'DrugBank ID' and 'CYPs'.
    cyp_list : list of str
        All possible CYP isoform names used to build the label matrix.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix indexed by molecule ID.
    Y : pd.DataFrame
        Binary label matrix (molecules × CYPs), indexed by molecule ID.
    """
    repr_df = pd.read_csv(fp_path, sep="\t", index_col=0)
    print(f"Loaded representation: {repr_df.shape}")

    labels_df = pd.read_csv(labels_path, sep=",")
    Y = pd.DataFrame(0, index=labels_df["DrugBank ID"], columns=cyp_list)

    for _, row in labels_df.iterrows():
        mol_id = row["DrugBank ID"]
        try:
            cyps = ast.literal_eval(str(row["CYPs"]))
        except (ValueError, SyntaxError):
            cyps = []
            print(
                f"Warning: Could not parse CYPs for molecule {mol_id}. "
                "Setting to empty list."
            )
        for c in cyps:
            if c in Y.columns:
                Y.loc[mol_id, c] = 1

    common_ids = repr_df.index.intersection(Y.index)
    X = repr_df.loc[common_ids]
    Y = Y.loc[common_ids]
    print(f"Molecules with both representation and labels: {len(common_ids)}")
    return X, Y

def train_and_evaluate_cyp_with_thresholds(X_train, y_train, X_val, y_val, xgb_params, thresholds, scale_pos_weight):
    """
    Train a single XGBoost model and evaluate multiple thresholds on validation fold.
    Returns a list of MCCs (one per threshold) in the same order as thresholds.
    """
    full_params = {**XGB_FIXED, **xgb_params, 'scale_pos_weight': scale_pos_weight}
    model = xgb.XGBClassifier(**full_params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]

    mccs = []
    for th in thresholds:
        pred = (proba >= th).astype(int)
        try:
            mcc = matthews_corrcoef(y_val, pred)
        except Exception:
            mcc = 0.0
        mccs.append(mcc)
    return mccs

def cross_validate_cyp_with_thresholds(X, y, folds, xgb_params, thresholds, scale_pos_weight):
    """
    Perform CV for one CYP using pre-computed fold indices.
    For each threshold, collect MCC from all folds, then average.
    Returns:
        best_threshold (float) - threshold with highest average MCC (first if tie).
        best_avg_mcc (float) - corresponding average MCC.
    """
    # For each threshold, accumulate MCCs across folds
    mcc_accum = {th: [] for th in thresholds}
    for fold_info in folds:
        train_ids_fold = fold_info["train_ids"]
        val_ids_fold = fold_info["val_ids"]

        X_train_f = X.loc[train_ids_fold]
        X_val_f = X.loc[val_ids_fold]
        y_train_f = y.loc[train_ids_fold]  
        y_val_f = y.loc[val_ids_fold]

        fold_mccs = train_and_evaluate_cyp_with_thresholds(
            X_train_f, y_train_f, X_val_f, y_val_f, xgb_params, thresholds, scale_pos_weight
        )
        for th, mcc_val in zip(thresholds, fold_mccs):
            mcc_accum[th].append(mcc_val)

    avg_mcc = {th: np.mean(mcc_accum[th]) for th in thresholds}
    best_th = max(avg_mcc.items(), key=lambda item: (item[1], -thresholds.index(item[0])))[0]
    best_mcc = avg_mcc[best_th]
    return best_th, best_mcc

def grid_search_over_cyps(X_train, Y_train, cyp_list, folds, param_grid, thresholds, scale_pos_weight_dict):
    """
    Exhaustive grid search over hyperparameters.
    For each combination:
        - For each CYP, find best threshold and its average validation MCC.
        - Macro MCC = mean of those best MCCs across CYPs.
    Returns:
        results_df (pd.DataFrame) - summary of all combos.
        best_params (dict) - best hyperparameters (learning_rate, max_depth, n_estimators).
        best_thresholds_per_cyp (dict) - for the best hyperparameters, stores best threshold for each CYP.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))
    results = []

    for combo_idx, combo in enumerate(combos, 1):
        xgb_params = dict(zip(keys, combo))   # no threshold here
        print(f"\n[{combo_idx}/{len(combos)}] Testing params: {xgb_params}")

        per_cyp_best_mcc = []
        per_cyp_best_th = []

        for cyp in cyp_list:
            y_cyp = Y_train[cyp]
            if y_cyp.sum() == 0:
                per_cyp_best_mcc.append(0.0)
                per_cyp_best_th.append(0.5)   # placeholder
                print(f"  {cyp}: no positives → MCC=0.0")
                continue

            best_th, best_mcc = cross_validate_cyp_with_thresholds(
                X_train, y_cyp, folds, xgb_params, thresholds, scale_pos_weight_dict[cyp]
            )
            per_cyp_best_mcc.append(best_mcc)
            per_cyp_best_th.append(best_th)
            print(f"  {cyp}: best th={best_th:.1f}, avg MCC={best_mcc:.4f}")

        macro_mcc = np.mean(per_cyp_best_mcc) if per_cyp_best_mcc else 0.0
        results.append({
            "params": xgb_params,
            "macro_mcc": macro_mcc,
            "per_cyp_mcc": per_cyp_best_mcc,
            "per_cyp_thresholds": per_cyp_best_th,
            "cyp_order": cyp_list
        })
        print(f"  -> Macro MCC = {macro_mcc:.4f}")

    # Now find the best hyperparameter set (using macro MCC, tie-break by variance of per-CYP MCCs)
    best_macro = max(r["macro_mcc"] for r in results)
    candidates = [r for r in results if abs(r["macro_mcc"] - best_macro) < 1e-6]
    if len(candidates) > 1:
        for cand in candidates:
            cand["variance"] = np.var(cand["per_cyp_mcc"])
        best_candidate = min(candidates, key=lambda x: (x["variance"], candidates.index(x)))
    else:
        best_candidate = candidates[0]

    best_params = best_candidate["params"]
    print(f"\nBest global parameters: {best_params} (macro MCC = {best_macro:.4f})")

    # For these best parameters, retrieve the per-CYP thresholds (we already stored them when we saw the best macro)
    # But careful: the best_candidate may not be the same as the first time we updated best_overall_thresholds
    # because of tie-breaking. So we extract from the best_candidate directly.
    best_thresholds_per_cyp = dict(zip(cyp_list, best_candidate["per_cyp_thresholds"]))

    # Convert results to DataFrame for saving
    results_df = pd.DataFrame([
        {"param_set": idx, "params": str(r["params"]), "macro_mcc": r["macro_mcc"]}
        for idx, r in enumerate(results)
    ])
    return results_df, best_params, best_thresholds_per_cyp


def retrain_all_cyps(X_train, Y_train, cyp_list, best_xgb_params, output_dir, scale_pos_weight_dict):
    """
    Train final model for each CYP on full training set using the best global
    XGBoost hyperparameters (no threshold in training). Saves each model as .joblib.
    Returns: dict of trained models.
    """
    models = {}
    for cyp in cyp_list:
        print(f"Retraining {cyp} on full 80% set...")
        y_cyp = Y_train[cyp]
        if y_cyp.sum() == 0:
            warnings.warn(f"CYP {cyp} has no positives in training - skipping model.")
            continue
        full_params = {**XGB_FIXED, **best_xgb_params, "scale_pos_weight": scale_pos_weight_dict[cyp]}
        model = xgb.XGBClassifier(**full_params)
        model.fit(X_train, y_cyp)
        models[cyp] = model
        joblib.dump(model, os.path.join(output_dir, f"{cyp}.joblib"))
    return models

def evaluate_test_set(
    models: dict,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    cyp_list: list,
    threshold_dict: dict,
) -> tuple:
    """
    Evaluate all per-CYP models on the held-out test set using per-CYP thresholds.

    Only CYPs present in `models` contribute to aggregate multi-label metrics,
    ensuring all three aggregate metrics are computed on the same subset.

    Parameters
    ----------
    models         : dict mapping CYP name to a trained classifier.
    X_test         : test feature matrix.
    Y_test         : true binary label matrix for all CYPs.
    cyp_list       : ordered list of CYP names.
    threshold_dict : dict mapping CYP name to its decision threshold.

    Returns
    -------
    per_cyp_mcc : dict mapping CYP name to its test MCC.
    macro_mcc   : float, mean MCC across evaluated CYPs.
    micro_f1    : float, micro-averaged F1 across evaluated CYPs.
    ham_loss    : float, Hamming loss across evaluated CYPs.
    """
    evaluated_cyps = []
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
        threshold = threshold_dict.get(cyp, 0.5)
        y_pred = (y_pred_prob >= threshold).astype(int)

        evaluated_cyps.append(cyp)
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except Exception:
            mcc = 0.0
        per_cyp_mcc[cyp] = mcc

    if not evaluated_cyps:
        return per_cyp_mcc, 0.0, 0.0, 1.0

    Y_pred_matrix = np.column_stack(y_pred_list)
    Y_true_matrix = Y_test[evaluated_cyps].values

    micro_f1 = f1_score(Y_true_matrix, Y_pred_matrix, average="micro")
    ham_loss = hamming_loss(Y_true_matrix, Y_pred_matrix)
    macro_mcc = np.mean(list(per_cyp_mcc.values()))

    return per_cyp_mcc, macro_mcc, micro_f1, ham_loss


def main(repr_file: str):
    print("=" * 60)
    print("Multi-CYP XGBoost training pipeline (threshold optimised per CYP)")
    print("=" * 60)

    # 1. Load data 
    repr_path = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "representations", repr_file))
    MODELS_DIR = os.path.join(OUTPUT_DIR, repr_file.split("_")[0])
    SAVE_DIR = os.path.join(SCRIPT_DIR, "results", "xgboost", repr_file.split("_")[0])
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    X, Y = load_data(repr_path, LABELS_FILE, cyp_labels)
    print(f"Total molecules: {X.shape[0]}, CYP classes: {Y.shape[1]}")

    # 2. Load pre-computed splits
    print("\nLoading frozen benchmark splits...")
    split_data = joblib.load(SPLITS_PATH)

    train_ids = split_data["train_ids"]
    test_ids = split_data["test_ids"]
    folds = split_data["folds"]
    used_cyps = split_data["used_cyps"]

    print(f"Loaded {len(train_ids)} train molecules")
    print(f"Loaded {len(test_ids)} test molecules")
    print(f"Loaded {len(folds)} CV folds")

    # 3. Apply test/train split to X and Y
    X_train = X.loc[train_ids]
    X_test = X.loc[test_ids]

    Y_train = Y.loc[train_ids]
    Y_test = Y.loc[test_ids]

    # 4. Precompute scale_pos_weight per CYP (negatives / positives in training set)
    cyp_scale_pos_weight = {}
    for cyp in used_cyps:
        pos = Y_train[cyp].sum()
        neg = len(Y_train) - pos
        if pos > 0:
            scale = neg / pos
        else:
            scale = 1.0  # fallback (should not happen because used_cyps have ≥10 positives)
        cyp_scale_pos_weight[cyp] = scale
        print(f"{cyp}: scale_pos_weight = {scale:.2f}")

    # 6. Grid search over hyperparameters (threshold optimised inside CV)
    results_df, best_params, best_thresholds = grid_search_over_cyps(
        X_train, Y_train, used_cyps, folds, param_grid, THRESHOLDS_TO_TRY, cyp_scale_pos_weight
    )

    # Save CV results and best parameters
    results_df.to_csv(os.path.join(SAVE_DIR, f"cv_results_{repr_file.split('_')[0]}.csv"), index=False)
    with open(os.path.join(SAVE_DIR, f"best_params_{repr_file.split('_')[0]}.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    with open(os.path.join(SAVE_DIR, f"best_thresholds_per_cyp_{repr_file.split('_')[0]}.json"), "w") as f:
        # convert numpy floats to Python floats for JSON
        thresholds_json = {k: float(v) for k, v in best_thresholds.items()}
        json.dump(thresholds_json, f, indent=2)

    # 7. Retrain models on full 80% using best XGBoost parameters 
    models = retrain_all_cyps(X_train, Y_train, used_cyps, best_params, MODELS_DIR, cyp_scale_pos_weight)

    # 8. Final evaluation on test set using per-CYP best thresholds
    per_cyp_mcc, macro_mcc, micro_f1, ham_loss = evaluate_test_set(
        models, X_test, Y_test, used_cyps, best_thresholds
    )

    # 9. Save test metrics
    test_metrics = {
        "macro_mcc": macro_mcc,
        "micro_f1": micro_f1,
        "hamming_loss": ham_loss,
        "per_cyp_mcc": {k: float(v) for k, v in per_cyp_mcc.items()},
    }
    with open(os.path.join(SAVE_DIR, f"test_metrics_{repr_file.split('_')[0]}.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # 10. Print final summary
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS (20% unseen data, per-CYP optimal thresholds)")
    print(f"Macro MCC      : {macro_mcc:.4f}")
    print(f"Micro F1       : {micro_f1:.4f}")
    print(f"Hamming loss   : {ham_loss:.4f}")
    print("\nPer-CYP MCC and used threshold:")
    for cyp in used_cyps:
        th = best_thresholds.get(cyp, 0.5)
        print(f"  {cyp}: threshold={th:.1f}, MCC={per_cyp_mcc[cyp]:.4f}")
    print("=" * 60)
    print(f"All models and results from {repr_file} saved in: {SAVE_DIR}")


if __name__ == "__main__":

    repr_files = [
        "MolE_output_representation.tsv",
        "chemberta_output_representation.tsv",
        "inchi_output_representation.tsv"
    ]
    for repr_file in repr_files:
        print(f"\n\n=== Processing representation: {repr_file} ===")
        main(repr_file)
    
    """repr_files = [
        "morgan_output_representation.tsv",
        "MolE_output_representation.tsv",
        "chemberta_output_representation.tsv",
        "inchi_output_representation.tsv"
    ]"""