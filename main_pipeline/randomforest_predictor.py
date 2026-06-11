"""
Multi-CYP classification using binary relevance (one RandomForestClassifier per CYP).

Global hyperparameter search over Random Forest parameters.
For each CYP and each hyperparameter set, threshold is optimised during CV
(by maximising average validation MCC over 0.1, 0.3, 0.5, 0.7).

Final evaluation on untouched 20% test set uses per-CYP optimal thresholds.

Methodology is intentionally parallel to:
- xgboost_base_predictor.py
- logistic_regression_baseline.py

This ensures a fair benchmark comparison across molecular representations
and ML algorithms.
"""

import os
import json
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import joblib

# CONFIGURATION
RANDOM_SEED = 33

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

LABELS_FILE = os.path.normpath(
    os.path.join(REPO_ROOT, "DataBases", "DrugBank_curated_df.csv")
)
SPLITS_PATH = os.path.join(SCRIPT_DIR, "splits", "benchmark_splits.joblib")

RESULTS_BASE = os.path.join(SCRIPT_DIR, "results", "random_forest")
MODELS_BASE = os.path.join(SCRIPT_DIR, "models", "random_forest")

cyp_labels = [
    "CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "CYP3A5",
    "CYP2C19", "CYP2C8", "CYP2B6", "CYP3A7", "CYP2E1",
    "CYP1A1", "CYP2A6", "CYP3A43",
]

THRESHOLDS_TO_TRY = [0.1, 0.3, 0.5, 0.7]

# RF HYPERPARAMETER GRID
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [12, 20],          # 'None' would risk overfitting, especially on unfrequent CYPs
    "min_samples_split": [3, 5],
    "max_features": ["sqrt", "log2"],
}

# FIXED RF PARAMETERS
RF_FIXED = {
    "class_weight": "balanced",          # handles imbalance (parallel to scale_pos_weight)
    "random_state": RANDOM_SEED,
    "n_jobs": 1,                         # reproducibility
}

# IMPORT SHARED FUNCTIONS
from xgboost_base_predictor import load_data
from xgboost_base_predictor import evaluate_test_set

# TRAINING & THRESHOLD EVALUATION
def train_and_evaluate_cyp_with_thresholds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    rf_params: dict,
    thresholds: list,
) -> list:
    """
    Train a single RandomForestClassifier and evaluate multiple thresholds
    on the validation fold.
    Returns list of MCCs (one per threshold).
    """
    model = RandomForestClassifier(**rf_params)
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

# CROSS-VALIDATION (one CYP, all folds)
def cross_validate_cyp_with_thresholds(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list,
    rf_params: dict,
    thresholds: list,
) -> tuple:
    """
    Perform CV for one CYP using pre‑computed fold indices.
    Returns (best_threshold, best_avg_mcc).
    """
    mcc_accum = {th: [] for th in thresholds}
    for fold_info in folds:
        train_ids_f = fold_info["train_ids"]
        val_ids_f = fold_info["val_ids"]

        X_train_f = X.loc[train_ids_f]
        X_val_f = X.loc[val_ids_f]
        y_train_f = y.loc[train_ids_f]
        y_val_f = y.loc[val_ids_f]

        fold_mccs = train_and_evaluate_cyp_with_thresholds(
            X_train_f, y_train_f, X_val_f, y_val_f, rf_params, thresholds
        )
        for th, mcc_val in zip(thresholds, fold_mccs):
            mcc_accum[th].append(mcc_val)

    avg_mcc = {th: np.mean(mcc_accum[th]) for th in thresholds}
    best_th = max(avg_mcc.items(), key=lambda item: (item[1], -thresholds.index(item[0])))[0]
    best_mcc = avg_mcc[best_th]
    return best_th, best_mcc

# GRID SEARCH OVER ALL CYPS
def grid_search_over_cyps(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    cyp_list: list,
    folds: list,
    param_grid: dict,
    thresholds: list,
) -> tuple:
    """
    Exhaustive grid search. For each hyperparameter combination:
        - For each CYP, find best threshold and its average MCC.
        - Macro MCC = mean of those best MCCs.
    Returns (results_df, best_params, best_thresholds_per_cyp).
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))
    results = []

    for combo_idx, combo in enumerate(combos, 1):
        rf_params = {**dict(zip(keys, combo)), **RF_FIXED}
        print(f"\n[{combo_idx}/{len(combos)}] Testing params: {rf_params}")

        per_cyp_best_mcc = []
        per_cyp_best_th = []

        for cyp in cyp_list:
            y_cyp = Y_train[cyp]
            if y_cyp.sum() == 0:
                per_cyp_best_mcc.append(0.0)
                per_cyp_best_th.append(0.5)
                print(f"  {cyp}: no positives → MCC=0.0")
                continue

            best_th, best_mcc = cross_validate_cyp_with_thresholds(
                X_train, y_cyp, folds, rf_params, thresholds
            )
            per_cyp_best_mcc.append(best_mcc)
            per_cyp_best_th.append(best_th)
            print(f"  {cyp}: best th={best_th:.1f}, avg MCC={best_mcc:.4f}")

        macro_mcc = np.mean(per_cyp_best_mcc) if per_cyp_best_mcc else 0.0
        results.append({
            "params": rf_params,
            "macro_mcc": macro_mcc,
            "per_cyp_mcc": per_cyp_best_mcc,
            "per_cyp_thresholds": per_cyp_best_th,
            "cyp_order": cyp_list,
        })
        print(f"  -> Macro MCC = {macro_mcc:.4f}")

    # Select best hyperparameter set (max macro MCC, tie‑break by variance)
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

    best_thresholds_per_cyp = dict(zip(cyp_list, best_candidate["per_cyp_thresholds"]))

    results_df = pd.DataFrame([
        {"param_set": idx, "params": str(r["params"]), "macro_mcc": r["macro_mcc"]}
        for idx, r in enumerate(results)
    ])
    return results_df, best_params, best_thresholds_per_cyp

# RETRAIN FINAL MODELS ON FULL 80% TRAINING SET
def retrain_all_cyps(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    cyp_list: list,
    best_rf_params: dict,
    output_dir: str,
) -> dict:
    """
    Train final RandomForestClassifier per CYP on full 80% training set.
    Saves each model as .joblib.
    """
    models = {}
    for cyp in cyp_list:
        print(f"Retraining {cyp} on full 80% set...")
        y_cyp = Y_train[cyp]
        if y_cyp.sum() == 0:
            warnings.warn(f"CYP {cyp} has no positives in training - skipping model.")
            continue
        model = RandomForestClassifier(**best_rf_params)
        model.fit(X_train, y_cyp)
        models[cyp] = model
        joblib.dump(model, os.path.join(output_dir, f"{cyp}.joblib"))
    return models

def main(repr_file: str) -> None:
    print("=" * 60)
    print("Multi-CYP Random Forest pipeline (threshold optimised per CYP)")
    print("=" * 60)

    # 1. Paths
    repr_name = repr_file.split("_")[0]
    repr_path = os.path.normpath(os.path.join(REPO_ROOT, "representations", repr_file))
    results_dir = os.path.join(RESULTS_BASE, repr_name)
    models_dir = os.path.join(MODELS_BASE, repr_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 2. Load data
    X, Y = load_data(repr_path, LABELS_FILE, cyp_labels)
    print(f"Total molecules: {X.shape[0]}, CYP classes: {Y.shape[1]}")

    # 3. Load frozen splits
    print("\nLoading frozen benchmark splits...")
    split_data = joblib.load(SPLITS_PATH)
    train_ids = split_data["train_ids"]
    test_ids = split_data["test_ids"]
    folds = split_data["folds"]
    used_cyps = split_data["used_cyps"]

    print(f"Loaded {len(train_ids)} train molecules")
    print(f"Loaded {len(test_ids)} test molecules")
    print(f"Loaded {len(folds)} CV folds")

    # 4. Apply splits
    X_train = X.loc[train_ids]
    X_test = X.loc[test_ids]
    Y_train = Y.loc[train_ids]
    Y_test = Y.loc[test_ids]

    # 5. Grid search (threshold inside CV)
    results_df, best_params, best_thresholds = grid_search_over_cyps(
        X_train, Y_train, used_cyps, folds, param_grid, THRESHOLDS_TO_TRY
    )

    # 6. Save CV results and best parameters (parallel naming)
    results_df.to_csv(os.path.join(results_dir, f"cv_results_{repr_file.split('_')[0]}.csv"), index=False)
    with open(os.path.join(results_dir, f"best_params_{repr_file.split('_')[0]}.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    with open(os.path.join(results_dir, f"best_thresholds_per_cyp_{repr_file.split('_')[0]}.json"), "w") as f:
        thresholds_json = {k: float(v) for k, v in best_thresholds.items()}
        json.dump(thresholds_json, f, indent=2)

    # 7. Retrain final models
    models = retrain_all_cyps(X_train, Y_train, used_cyps, best_params, models_dir)

    # 8. Test evaluation
    per_cyp_mcc, macro_mcc, micro_f1, ham_loss = evaluate_test_set(
        models, X_test, Y_test, used_cyps, best_thresholds
    )

    # 9. Save test metrics (parallel naming)
    test_metrics = {
        "macro_mcc": macro_mcc,
        "micro_f1": micro_f1,
        "hamming_loss": ham_loss,
        "per_cyp_mcc": {k: float(v) for k, v in per_cyp_mcc.items()},
    }
    with open(os.path.join(results_dir, f"test_metrics_{repr_file.split('_')[0]}.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # 10. Print summary
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
    print(f"All models and results from {repr_file} saved in: {results_dir}")

if __name__ == "__main__":
    repr_files = [
        "morgan_output_representation.tsv",
        "MolE_output_representation.tsv",
        "chemberta_output_representation.tsv",
        "inchi_output_representation.tsv",
    ]
    for repr_file in repr_files:
        print(f"\n\n=== Processing representation: {repr_file} ===")
        main(repr_file)