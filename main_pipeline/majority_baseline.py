"""
Majority-class baseline for multi-label CYP450 prediction.

This baseline predicts, for each CYP independently,
the majority class observed in the TRAINING SET.

Purpose:
- establish a naive benchmark
- quantify how much improvement learned models provide
- control for severe class imbalance

Uses the exact same frozen benchmark splits as all other models.
"""

import os
import json
import warnings
import ast
import joblib

import numpy as np
import pandas as pd

from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    hamming_loss
)

# CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SPLITS_PATH = os.path.join(
    SCRIPT_DIR,
    "splits",
    "benchmark_splits.joblib"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "models",
    "majority_baseline"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CYP_LABELS = [
    "CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "CYP3A5",
    "CYP2C19", "CYP2C8", "CYP2B6", "CYP3A7", "CYP2E1",
    "CYP1A1", "CYP2A6", "CYP3A43"
]


def load_labels(labels_path, cyp_list):
    """
    Load CYP labels and convert into binary matrix.

    Returns:
        Y (pd.DataFrame)
    """

    labels_df = pd.read_csv(labels_path)

    Y = pd.DataFrame(
        0,
        index=labels_df["DrugBank ID"],
        columns=cyp_list
    )

    for _, row in labels_df.iterrows():

        mol_id = row["DrugBank ID"]

        try:
            cyps = ast.literal_eval(str(row["CYPs"]))
        except (ValueError, SyntaxError):
            cyps = []

        for c in cyps:
            if c in Y.columns:
                Y.loc[mol_id, c] = 1

    return Y


def determine_majority_classes(Y_train, used_cyps):
    """
    Determine majority class for each CYP.

    Returns:
        dict:
            {
                "CYP3A4": 1,
                "CYP2D6": 0,
                ...
            }
    """

    majority_dict = {}

    for cyp in used_cyps:

        positives = Y_train[cyp].sum()
        negatives = len(Y_train) - positives

        if positives > negatives:
            majority_class = 1
        else:
            majority_class = 0

        majority_dict[cyp] = majority_class

        print(
            f"{cyp}: "
            f"positives={positives}, "
            f"negatives={negatives}, "
            f"majority_class={majority_class}"
        )

    return majority_dict


def evaluate_majority_baseline(
    majority_dict,
    Y_test,
    used_cyps
):
    """
    Evaluate majority baseline on test set.
    """

    per_cyp_mcc = {}

    y_true_all = []
    y_pred_all = []

    for cyp in used_cyps:

        y_true = Y_test[cyp].values

        majority_prediction = majority_dict[cyp]

        y_pred = np.full_like(
            y_true,
            fill_value=majority_prediction
        )

        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except Exception:
            mcc = 0.0

        per_cyp_mcc[cyp] = mcc

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        print(f"{cyp}: MCC = {mcc:.4f}")

    # Convert into matrices
    Y_true_matrix = np.column_stack(y_true_all)
    Y_pred_matrix = np.column_stack(y_pred_all)

    macro_mcc = np.mean(list(per_cyp_mcc.values()))

    micro_f1 = f1_score(
        Y_true_matrix,
        Y_pred_matrix,
        average="micro"
    )

    ham_loss = hamming_loss(
        Y_true_matrix,
        Y_pred_matrix
    )

    return (
        per_cyp_mcc,
        macro_mcc,
        micro_f1,
        ham_loss
    )


def main():

    print("=" * 60)
    print("Majority-class baseline")
    print("=" * 60)

    # MODIFY PATH IF NEEDED
    labels_file = "DrugBank_curated_df.csv"

    # 1. Load labels
    Y = load_labels(
        labels_file,
        CYP_LABELS
    )

    print(f"Loaded labels: {Y.shape}")

    # 2. Load frozen benchmark splits
    split_data = joblib.load(SPLITS_PATH)

    train_ids = split_data["train_ids"]
    test_ids = split_data["test_ids"]

    used_cyps = split_data["used_cyps"]

    print(f"Training molecules: {len(train_ids)}")
    print(f"Test molecules:     {len(test_ids)}")

    print(f"Used CYPs: {used_cyps}")

    # 3. Reconstruct train/test label matrices
    Y_train = Y.loc[train_ids]
    Y_test = Y.loc[test_ids]

    # 4. Determine majority class per CYP
    majority_dict = determine_majority_classes(
        Y_train,
        used_cyps
    )

    # Save majority classes
    with open(
        os.path.join(OUTPUT_DIR, "majority_classes.json"),
        "w"
    ) as f:
        json.dump(majority_dict, f, indent=4)

    # 5. Evaluate baseline
    (
        per_cyp_mcc,
        macro_mcc,
        micro_f1,
        ham_loss
    ) = evaluate_majority_baseline(
        majority_dict,
        Y_test,
        used_cyps
    )

    # 6. Save results
    results = {
        "macro_mcc": macro_mcc,
        "micro_f1": micro_f1,
        "hamming_loss": ham_loss,
        "per_cyp_mcc": per_cyp_mcc
    }

    with open(
        os.path.join(OUTPUT_DIR, "majority_baseline_results.json"),
        "w"
    ) as f:
        json.dump(results, f, indent=4)

    # 7. Final report
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"Macro MCC:   {macro_mcc:.4f}")
    print(f"Micro F1:    {micro_f1:.4f}")
    print(f"HammingLoss: {ham_loss:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()