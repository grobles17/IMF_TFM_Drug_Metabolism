"""
Generate and freeze:
- multilabel-preserving 80/20 train/test split
- multilabel-preserving CV folds on the training set

The generated splits are saved permanently and reused by all models
(XGBoost, Logistic Regression, Random Forest, etc.) and all molecular
representations (ECFP6, MolE, ChemBERTa, InChI).

This ensures:
- reproducibility
- fair benchmark comparison
- identical molecule partitions across experiments
"""

import os
import json
import warnings
import ast
import joblib

import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import (
    MultilabelStratifiedShuffleSplit,
    MultilabelStratifiedKFold
)

# CONFIGURATION
RANDOM_SEED = 33
TEST_FRACTION = 0.2
CV_FOLDS = 3
MIN_POS_PER_FOLD = 10
MIN_POSITIVES = 5

# List of all CYP classes
CYP_LABELS = [
    "CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "CYP3A5",
    "CYP2C19", "CYP2C8", "CYP2B6", "CYP3A7", "CYP2E1",
    "CYP1A1", "CYP2A6", "CYP3A43"
]

# PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(SCRIPT_DIR, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

REPRESENTATION_FILE = os.path.normpath(
    os.path.join(
        SCRIPT_DIR, 
        "..",
        "representations", 
        "morgan_output_representation.tsv"))
LABELS_FILE = os.path.normpath(
    os.path.join(
        SCRIPT_DIR, 
        "..",
        "DataBases", 
        "DrugBank_curated_df.csv"))

def load_data(fp_path, labels_path, cyp_list):
    """
    Load fingerprints and CYP labels.

    Args:
        fp_path (str): TSV file with ID in first column, then fingerprint counts.
        labels_path (str): CSV file with columns 'DrugBank ID' and 'CYPs'.
        cyp_list (list): All possible CYP names.

    Returns:
        X (pd.DataFrame): Fingerprints, index = molecule ID.
        Y (pd.DataFrame): Binary matrix (molecules x CYPs), index = same ID.
    """
    repr_df = pd.read_csv(fp_path, sep="\t", index_col=0)
    print(f"Loaded representation: {repr_df.shape}")

    labels_df = pd.read_csv(labels_path, sep=",")
    all_cyps = cyp_list
    Y = pd.DataFrame(0, 
                     index=labels_df["DrugBank ID"], 
                     columns=all_cyps)

    for _, row in labels_df.iterrows():
        mol_id = row["DrugBank ID"]
        try:
            cyps = ast.literal_eval(str(row["CYPs"]))
        except (ValueError, SyntaxError):
            cyps = []
            print(f"Warning: Could not parse CYPs for molecule {mol_id}. Setting to empty list.")
        for c in cyps:
            if c in Y.columns:
                Y.loc[mol_id, c] = 1

    common_ids = repr_df.index.intersection(Y.index)
    X = repr_df.loc[common_ids]
    Y = Y.loc[common_ids]
    print(f"Molecules with both representation and labels: {len(common_ids)}")
    return X, Y

def iterative_stratified_split(X, Y, test_size=0.2, random_state=33):
    """
    Multilabel-preserving train/test split.

    Returns:
        train_ids (list)
        test_ids (list)
    """

    X_arr = X.to_numpy()
    Y_arr = Y.to_numpy()

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, test_idx = next(msss.split(X_arr, Y_arr))

    train_ids = X.index[train_idx].tolist()
    test_ids = X.index[test_idx].tolist()

    return train_ids, test_ids

def check_cyp_distribution(Y_train, Y_test, cyp_list, min_positives=10):
    """
    Check train/test CYP distributions. 
    Warn if any CYP has too few positives in train or test.

    Returns:
        safe_cyps (list)
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


def create_cv_folds(X_train, Y_train, used_cyps):
    """
    Create multilabel-stratified CV folds.

    Ensures every validation fold has at least
    MIN_POS_PER_FOLD positives for every CYP.

    Returns:
        fold_data (list of dicts)
        final_seed (int)
    """

    seed = RANDOM_SEED

    while True:

        mskf = MultilabelStratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=seed
        )

        folds = list(mskf.split(X_train, Y_train))

        ok = True
        for cyp in used_cyps:
            for fold_idx, (_, val_idx) in enumerate(folds):
                pos = Y_train.iloc[val_idx][cyp].sum()
                if pos < MIN_POS_PER_FOLD:
                    print(
                        f"Seed {seed}: "
                        f"fold {fold_idx+1} for {cyp} "
                        f"has only {pos} positives "
                        f"(<{MIN_POS_PER_FOLD})"
                    )
                    ok = False
                    break
            if not ok:
                break

        if ok:
            print(
                f"All folds valid with seed={seed} "
                f"(≥{MIN_POS_PER_FOLD} positives per CYP)"
            )
            break
        print(f"Retrying with seed={seed+1}...")
        seed += 1

    fold_data = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):

        fold_info = {
            "fold": fold_idx + 1,
            "train_ids": X_train.index[train_idx].tolist(),
            "val_ids": X_train.index[val_idx].tolist()
        }

        fold_data.append(fold_info)

    return fold_data, seed

# MAIN
def main():

    print("=" * 60)
    print("Creating frozen benchmark splits")
    print("=" * 60)

    # Load data
    X, Y = load_data(
        REPRESENTATION_FILE,
        LABELS_FILE,
        CYP_LABELS
    )

    print(f"Total molecules: {len(X)}")

    # Train/test split
    train_ids, test_ids = iterative_stratified_split(
        X,
        Y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED
    )

    X_train = X.loc[train_ids]
    X_test = X.loc[test_ids]

    Y_train = Y.loc[train_ids]
    Y_test = Y.loc[test_ids]

    print(f"Training molecules: {len(train_ids)}")
    print(f"Test molecules:     {len(test_ids)}")

    # CYP filtering
    used_cyps = check_cyp_distribution(
        Y_train,
        Y_test,
        CYP_LABELS,
        min_positives=MIN_POSITIVES
    )

    print(f"Usable CYPs: {used_cyps}")

    if len(used_cyps) == 0:
        raise RuntimeError("No CYPs passed minimum positivity checks.")

    # CV folds
    fold_data, final_seed = create_cv_folds(
        X_train,
        Y_train,
        used_cyps
    )

    # Save splits
    split_data = {
        "random_seed": RANDOM_SEED,
        "final_cv_seed": final_seed,
        "test_fraction": TEST_FRACTION,
        "cv_folds": CV_FOLDS,
        "min_pos_per_fold": MIN_POS_PER_FOLD,
        "min_positives": MIN_POSITIVES,
        "used_cyps": used_cyps,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "folds": fold_data
    }

    json_path = os.path.join(SPLITS_DIR, "benchmark_splits.json")

    with open(json_path, "w") as f:
        json.dump(split_data, f, indent=4)

    print("\nSaved benchmark splits:")
    print(json_path)

    joblib.dump(
        split_data,
        os.path.join(SPLITS_DIR, "benchmark_splits.joblib")
    )

    print("\nDone.")

if __name__ == "__main__":
    main()