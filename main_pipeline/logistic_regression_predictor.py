"""
Multi-CYP classification using binary relevance (one LogisticRegression per CYP).
Global hyperparameter search over regularisation strength C and solver.
For each CYP and each hyperparameter set, threshold is optimised during CV
(by maximising average validation MCC over 0.1, 0.3, 0.5, 0.7).
Final evaluation on untouched 20% test set uses per-CYP optimal thresholds.

Methodology is identical to xgboost_base_predictor.py to allow fair comparison.
"""

import os
import json
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.exceptions import ConvergenceWarning
import joblib

# CONFIGURATION
RANDOM_SEED = 33
FAILED_CONVERGENCE_LOGS = []

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

LABELS_FILE = os.path.normpath(
    os.path.join(REPO_ROOT, "DataBases", "DrugBank_curated_df.csv")
)
SPLITS_PATH = os.path.join(SCRIPT_DIR, "splits", "benchmark_splits.joblib")

RESULTS_BASE = os.path.join(SCRIPT_DIR, "results", "logistic_regression")
MODELS_BASE = os.path.join(SCRIPT_DIR, "models", "logistic_regression")

# All CYP classes (must match xgboost_base_predictor.py)
cyp_labels = [
    "CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "CYP3A5",
    "CYP2C19", "CYP2C8", "CYP2B6", "CYP3A7", "CYP2E1",
    "CYP1A1", "CYP2A6", "CYP3A43",
]

# Thresholds to evaluate during CV (identical to XGBoost pipeline)
THRESHOLDS_TO_TRY = [0.1, 0.3, 0.5, 0.7]

# Hyperparameter grid.
# Notes on solver choice:
#   - "liblinear" supports L1 and L2, handles sparse/count matrices well,
#     and is fast for moderately sized datasets.
#   - "lbfgs" would only support L2 but was failing to converge on some 
#     CYPs even with high max_iter, so it was dropped.
#   - "saga" supports both L1 and L2, is robust to convergence issues, 
#     and can handle larger datasets, but is slower than liblinear for smaller ones. 
#   class_weight="balanced" replaces scale_pos_weight from XGBoost and handles imbalance.
param_grid = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": ["liblinear", "saga"],
    "penalty": ["l1","l2"],  
}

# Fixed parameters applied to every LogisticRegression instance
LR_FIXED = {
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "max_iter": 5000, #started at 1000, then 2000 but some models failed to converge
}
"""lbfgs was failing to converge on some CYPs even with max_iter=10 000, 
    so it was switched to liblinear and saga, with the added benefit
    of supporting L1 penalty and thus more sparsity in the final models.
"""
# DATA LOADING (identical to xgboost_base_predictor.py)
from xgboost_base_predictor import load_data

# TRAINING & THRESHOLD EVALUATION
def train_and_evaluate_cyp_with_thresholds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    lr_params: dict,
    thresholds: list,
) -> list:
    """
    Train a single LogisticRegression and evaluate multiple thresholds on the
    validation fold.

    Parameters
    ----------
    X_train, y_train : training features and binary labels for one CYP.
    X_val, y_val     : validation features and binary labels for one CYP.
    lr_params        : keyword arguments passed to LogisticRegression.
    thresholds       : list of float probability cut-offs to evaluate.

    Returns
    -------
    list of float
        MCC for each threshold in the same order as `thresholds`.
    """
    model = LogisticRegression(**lr_params)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        model.fit(X_train, y_train)

        for warn in caught_warnings:
            if issubclass(warn.category, ConvergenceWarning):

                FAILED_CONVERGENCE_LOGS.append({
                    "solver": lr_params.get("solver"),
                    "penalty": lr_params.get("penalty"),
                    "C": lr_params.get("C"),
                    "warning_message": str(warn.message),
                    "n_train_samples": len(X_train),
                    "n_features": X_train.shape[1],
                })
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


def cross_validate_cyp_with_thresholds(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list,
    lr_params: dict,
    thresholds: list,
) -> tuple:
    """
    Perform CV for one CYP using pre-computed fold indices.

    For each threshold, MCC values are collected across all folds and averaged.

    Parameters
    ----------
    X        : full training feature matrix (indexed by molecule ID).
    y        : binary label series for one CYP.
    folds    : list of dicts with keys 'train_ids' and 'val_ids'.
    lr_params: keyword arguments passed to LogisticRegression.
    thresholds: list of float probability cut-offs.

    Returns
    -------
    best_threshold : float
        Threshold with highest average MCC (ties broken by lower threshold index).
    best_avg_mcc   : float
        Corresponding average validation MCC.
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
            X_train_f, y_train_f, X_val_f, y_val_f, lr_params, thresholds
        )
        for th, mcc_val in zip(thresholds, fold_mccs):
            mcc_accum[th].append(mcc_val)

    avg_mcc = {th: np.mean(mcc_accum[th]) for th in thresholds}
    best_th = max(
        avg_mcc.items(),
        key=lambda item: (item[1], -thresholds.index(item[0])),
    )[0]
    best_mcc = avg_mcc[best_th]
    return best_th, best_mcc


# GRID SEARCH
def grid_search_over_cyps(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    cyp_list: list,
    folds: list,
    param_grid: dict,
    thresholds: list,
) -> tuple:
    """
    Exhaustive grid search over hyperparameters across all CYP isoforms.

    For each hyperparameter combination:
        - For each CYP, find the threshold that maximises average validation MCC.
        - Macro MCC = mean of those per-CYP best MCCs.

    Best overall hyperparameters are selected by highest macro MCC; ties are broken
    by lowest variance of per-CYP MCCs (i.e., prefer more consistent models).

    Parameters
    ----------
    X_train  : training feature matrix.
    Y_train  : binary label matrix for all CYPs.
    cyp_list : list of CYP isoform names to include.
    folds    : pre-computed CV fold definitions.
    param_grid : dict mapping hyperparameter names to lists of candidate values.
    thresholds : list of float probability cut-offs.

    Returns
    -------
    results_df            : pd.DataFrame summarising all combinations.
    best_params           : dict of best hyperparameters (C, solver + fixed params).
    best_thresholds_per_cyp : dict mapping CYP name to its optimal threshold.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))
    results = []

    for combo_idx, combo in enumerate(combos, 1):
        # Merge grid params with fixed params
        lr_params = {**dict(zip(keys, combo)), **LR_FIXED}
        print(f"\n[{combo_idx}/{len(combos)}] Testing params: {lr_params}")

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
                X_train, y_cyp, folds, lr_params, thresholds
            )
            per_cyp_best_mcc.append(best_mcc)
            per_cyp_best_th.append(best_th)
            print(f"  {cyp}: best th={best_th:.1f}, avg MCC={best_mcc:.4f}")

        macro_mcc = np.mean(per_cyp_best_mcc) if per_cyp_best_mcc else 0.0
        results.append({
            "params": lr_params,
            "macro_mcc": macro_mcc,
            "per_cyp_mcc": per_cyp_best_mcc,
            "per_cyp_thresholds": per_cyp_best_th,
            "cyp_order": cyp_list,
        })
        print(f"  -> Macro MCC = {macro_mcc:.4f}")

    # Select best hyperparameter set
    best_macro = max(r["macro_mcc"] for r in results)
    candidates = [r for r in results if abs(r["macro_mcc"] - best_macro) < 1e-6]
    if len(candidates) > 1:
        for cand in candidates:
            cand["variance"] = np.var(cand["per_cyp_mcc"])
        best_candidate = min(
            candidates, key=lambda x: (x["variance"], candidates.index(x))
        )
    else:
        best_candidate = candidates[0]

    best_params = best_candidate["params"]
    print(f"\nBest global parameters: {best_params} (macro MCC = {best_macro:.4f})")

    best_thresholds_per_cyp = dict(
        zip(cyp_list, best_candidate["per_cyp_thresholds"])
    )

    results_df = pd.DataFrame([
        {
            "param_set": idx,
            "params": str(r["params"]),
            "macro_mcc": r["macro_mcc"],
        }
        for idx, r in enumerate(results)
    ])
    return results_df, best_params, best_thresholds_per_cyp


# RETRAIN ON FULL TRAINING SET
def retrain_all_cyps(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    cyp_list: list,
    best_lr_params: dict,
    output_dir: str,
) -> dict:
    """
    Train a final LogisticRegression for each CYP on the full training set
    using the best global hyperparameters. Saves each model as a .joblib file.

    Parameters
    ----------
    X_train       : full training feature matrix.
    Y_train       : binary label matrix for all CYPs.
    cyp_list      : list of CYP isoform names.
    best_lr_params: hyperparameters selected by grid search (includes fixed params).
    output_dir    : directory where .joblib models are written.

    Returns
    -------
    dict mapping CYP name to its trained LogisticRegression model.
    """
    models = {}
    for cyp in cyp_list:
        print(f"Retraining {cyp} on full 80% set...")
        y_cyp = Y_train[cyp]
        if y_cyp.sum() == 0:
            warnings.warn(
                f"CYP {cyp} has no positives in training — skipping model."
            )
            continue
        model = LogisticRegression(**best_lr_params)
        model.fit(X_train, y_cyp)
        models[cyp] = model
        joblib.dump(model, os.path.join(output_dir, f"{cyp}.joblib"))
    return models


from xgboost_base_predictor import evaluate_test_set


# MAIN PIPELINE
def main(repr_file: str) -> None:
    """
    Run the full logistic regression pipeline for one molecular representation.

    Steps:
        1. Load features and labels.
        2. Load frozen benchmark splits.
        3. Perform hyperparameter grid search with CV threshold optimisation.
        4. Retrain final per-CYP models on full 80% training set.
        5. Evaluate on held-out 20% test set.
        6. Save all results and models to disk.

    Parameters
    ----------
    repr_file : str
        Filename of the representation TSV (e.g. 'morgan_output_representation.tsv').
        The stem before the first underscore is used as the output subdirectory name.
    """
    print("=" * 60)
    print("Multi-CYP Logistic Regression pipeline (threshold optimised per CYP)")
    print("=" * 60)

    # 1. Resolve paths and create output directories
    repr_name = repr_file.split("_")[0]   # e.g., "morgan", "MolE", "chemberta", "inchi"
    repr_path = os.path.normpath(
        os.path.join(REPO_ROOT, "representations", repr_file)
    )
    results_dir = os.path.join(RESULTS_BASE, repr_name)
    models_dir = os.path.join(MODELS_BASE, repr_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 2. Load data
    X, Y = load_data(repr_path, LABELS_FILE, cyp_labels)
    print(f"Total molecules: {X.shape[0]}, CYP classes: {Y.shape[1]}")

    # 3. Load frozen benchmark splits
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

    # 5. Grid search over hyperparameters (threshold optimised inside CV)
    results_df, best_params, best_thresholds = grid_search_over_cyps(
        X_train, Y_train, used_cyps, folds, param_grid, THRESHOLDS_TO_TRY
    )

    # Save CV results and best parameters
    results_df.to_csv(os.path.join(results_dir, f"cv_results_{repr_name.split('_')[0]}.csv"), index=False)
    # Serialise only the grid-searchable params (drop fixed params from JSON)
    grid_keys = list(param_grid.keys())
    best_grid_params = {k: best_params[k] for k in grid_keys}
    with open(os.path.join(results_dir, f"best_params_{repr_name.split('_')[0]}.json"), "w") as f:
        json.dump(best_grid_params, f, indent=2)
    with open(os.path.join(results_dir, f"best_thresholds_per_cyp_{repr_name.split('_')[0]}.json"), "w") as f:
        thresholds_json = {k: float(v) for k, v in best_thresholds.items()}
        json.dump(thresholds_json, f, indent=2)

    # 6. Retrain final models on full 80% training set
    models = retrain_all_cyps(X_train, Y_train, used_cyps, best_params, models_dir)

    # 7. Evaluate on held-out test set
    per_cyp_mcc, macro_mcc, micro_f1, ham_loss = evaluate_test_set(
        models, X_test, Y_test, used_cyps, best_thresholds
    )

    # 8. Save test metrics
    test_metrics = {
        "macro_mcc": macro_mcc,
        "micro_f1": micro_f1,
        "hamming_loss": ham_loss,
        "per_cyp_mcc": {k: float(v) for k, v in per_cyp_mcc.items()},
    }
    with open(os.path.join(results_dir, f"test_metrics_{repr_name.split('_')[0]}.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # Save convergence diagnostics
    if FAILED_CONVERGENCE_LOGS:
        convergence_df = pd.DataFrame(FAILED_CONVERGENCE_LOGS)

        convergence_df.to_csv(
            os.path.join(results_dir, "convergence_warnings.csv"),
            index=False
        )

        print(
            f"Saved {len(convergence_df)} convergence warnings."
        )

    # 9. Print final summary
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