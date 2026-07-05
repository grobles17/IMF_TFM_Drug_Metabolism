"""
collect_results.py

Loads all benchmark results from JSON/CSV files, organises them into
analysis-ready dataframes, and exports them as CSV files.
"""

import os
import json
import warnings
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

RESULTS_ROOT = Path("./main_pipeline/results")

# Models to process (majority_baseline is handled separately)
MODELS = ["logistic_regression", "random_forest", "xgboost_results"]
REPRESENTATIONS = ["MolE", "chemberta", "inchi", "morgan"]

# Suffix used in filenames (matches the representation name)
REPR_SUFFIX = {
    "MolE": "mole",        # adjust if actual files use "MolE"
    "chemberta": "chemberta",
    "inchi": "inchi",
    "morgan": "morgan"
}

# Majority baseline file
MAJORITY_FILE = RESULTS_ROOT / "majority_baseline" / "majority_baseline_results.json"

# Output directory
OUTPUT_DIR = RESULTS_ROOT / "collected_results"

# Fixed order of CYPs (by training frequency, from most frequent to least)
CYP_ORDER = [
    "CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "CYP3A5",
    "CYP2C19", "CYP2C8", "CYP2B6", "CYP3A7", "CYP2E1",
    "CYP1A1", "CYP2A6", "CYP3A43"
]


# ----------------------------------------------------------------------
# Helper functions: file finding and loading
# ----------------------------------------------------------------------

def find_files(model: str, repr_name: str) -> Dict[str, Optional[Path]]:
    """
    Locate the result files for a given (model, representation).

    Returns a dict with keys: 'thresholds', 'params', 'metrics', 'cv'
    or None if a file is missing.
    """
    base_dir = RESULTS_ROOT / model / repr_name
    suffix = REPR_SUFFIX[repr_name]

    files = {
        "thresholds": base_dir / f"best_thresholds_per_cyp_{suffix}.json",
        "params": base_dir / f"best_params_{suffix}.json",
        "metrics": base_dir / f"test_metrics_{suffix}.json",
        "cv": base_dir / f"cv_results_{suffix}.csv"
    }

    for key, path in files.items():
        if not path.exists():
            warnings.warn(f"File not found: {path}")
            files[key] = None

    return files


def load_thresholds(file_path: Path) -> pd.Series:
    """Load thresholds JSON and return as a Series (CYP -> threshold)."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.Series(data)


def load_params(file_path: Path) -> Dict:
    """Load best parameters JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_metrics(file_path: Path) -> Dict:
    """Load test metrics JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_cv(file_path: Path) -> pd.DataFrame:
    """Load cross‑validation results CSV."""
    return pd.read_csv(file_path)


def load_majority_baseline(file_path: Path) -> Dict:
    """Load majority baseline results JSON."""
    if not file_path.exists():
        warnings.warn(f"Majority baseline file not found: {file_path}")
        return None
    with open(file_path, 'r') as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Core collection functions
# ----------------------------------------------------------------------

def collect_experiment_metrics(model: str, repr_name: str) -> Optional[Dict]:
    """
    Load metrics and parameters for a single (model, representation).

    Returns a dict with:
        - 'metrics': dict with macro_mcc, micro_f1, hamming_loss, per_cyp_mcc
        - 'params': dict of best parameters
        - 'thresholds': Series of thresholds per CYP
        - 'cv_df': DataFrame of CV results (or None)
    or None if metrics file is missing.
    """
    files = find_files(model, repr_name)

    if files["metrics"] is None:
        return None

    metrics = load_metrics(files["metrics"])
    params = load_params(files["params"]) if files["params"] is not None else {}
    thresholds = load_thresholds(files["thresholds"]) if files["thresholds"] is not None else pd.Series()
    cv_df = load_cv(files["cv"]) if files["cv"] is not None else None

    return {
        "metrics": metrics,
        "params": params,
        "thresholds": thresholds,
        "cv_df": cv_df
    }


def collect_all_experiments() -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    """
    Iterate over all models and representations, collect data.

    Returns:
        - experiments: list of dicts with keys: model, representation, metrics, params, thresholds, cv_df
        - all_cv_df: concatenated CV results (with model and representation columns)
        - threshold_records: list of dicts (model, representation, cyp, threshold)
    """
    experiments = []
    all_cv_dfs = []
    threshold_records = []

    for model in MODELS:
        print(f"Loading {model}...")
        for repr_name in REPRESENTATIONS:
            data = collect_experiment_metrics(model, repr_name)
            if data is None:
                continue

            # Store experiment
            experiments.append({
                "model": model,
                "representation": repr_name,
                "metrics": data["metrics"],
                "params": data["params"],
                "thresholds": data["thresholds"],
                "cv_df": data["cv_df"]
            })

            # Collect thresholds
            for cyp, th in data["thresholds"].items():
                threshold_records.append({
                    "model": model,
                    "representation": repr_name,
                    "cyp": cyp,
                    "threshold": th
                })

            # Collect CV data
            if data["cv_df"] is not None:
                cv = data["cv_df"].copy()
                cv["model"] = model
                cv["representation"] = repr_name
                all_cv_dfs.append(cv)

    # Concatenate all CV dataframes
    if all_cv_dfs:
        all_cv_df = pd.concat(all_cv_dfs, ignore_index=True)
    else:
        all_cv_df = pd.DataFrame()

    thresholds_df = pd.DataFrame(threshold_records)

    return experiments, all_cv_df, thresholds_df


# ----------------------------------------------------------------------
# Build master and per‑CYP dataframes
# ----------------------------------------------------------------------

def build_master_dataframe(experiments: List[Dict]) -> pd.DataFrame:
    """
    Create master results table.

    Columns: model, representation, macro_mcc, micro_f1, hamming_loss, best_params
    """
    records = []
    for exp in experiments:
        metrics = exp["metrics"]
        # Extract required metrics (some may be missing)
        macro_mcc = metrics.get("macro_mcc", None)
        micro_f1 = metrics.get("micro_f1", None)
        hamming_loss = metrics.get("hamming_loss", None)
        # Store best_params as pretty JSON string (indent=2) for readability
        if exp["params"]:
            best_params = json.dumps(exp["params"], indent=2)
        else:
            best_params = None

        records.append({
            "model": exp["model"],
            "representation": exp["representation"],
            "macro_mcc": macro_mcc,
            "micro_f1": micro_f1,
            "hamming_loss": hamming_loss,
            "best_params": best_params
        })

    return pd.DataFrame(records)


def build_per_cyp_dataframe(experiments: List[Dict]) -> pd.DataFrame:
    """
    Unpack per_cyp_mcc dictionaries.

    Columns: model, representation, cyp, mcc
    """
    records = []
    for exp in experiments:
        metrics = exp["metrics"]
        per_cyp_mcc = metrics.get("per_cyp_mcc", {})
        for cyp, mcc in per_cyp_mcc.items():
            records.append({
                "model": exp["model"],
                "representation": exp["representation"],
                "cyp": cyp,
                "mcc": mcc
            })
    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# Summary tables (by representation, by model)
# ----------------------------------------------------------------------

def summarise_by_representation(master_df: pd.DataFrame) -> pd.DataFrame:
    """Group by representation and compute mean/std of metrics."""
    group = master_df.groupby("representation")
    summary = group.agg({
        "macro_mcc": ["mean", "std"],
        "micro_f1": ["mean", "std"],
        "hamming_loss": ["mean", "std"]
    }).round(4)
    # Flatten columns
    summary.columns = [
        "mean_macro_mcc", "std_macro_mcc",
        "mean_micro_f1", "std_micro_f1",
        "mean_hamming_loss", "std_hamming_loss"
    ]
    summary = summary.sort_values("mean_macro_mcc", ascending=False)
    return summary.reset_index()


def summarise_by_model(master_df: pd.DataFrame) -> pd.DataFrame:
    """Group by model and compute mean/std of metrics."""
    group = master_df.groupby("model")
    summary = group.agg({
        "macro_mcc": ["mean", "std"],
        "micro_f1": ["mean", "std"],
        "hamming_loss": ["mean", "std"]
    }).round(4)
    summary.columns = [
        "mean_macro_mcc", "std_macro_mcc",
        "mean_micro_f1", "std_micro_f1",
        "mean_hamming_loss", "std_hamming_loss"
    ]
    summary = summary.sort_values("mean_macro_mcc", ascending=False)
    return summary.reset_index()


# ----------------------------------------------------------------------
# Threshold analysis (with fixed CYP order)
# ----------------------------------------------------------------------

def analyse_thresholds(thresholds_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce two threshold tables:
        - summary by CYP: mean, std, min, max (ordered by CYP_ORDER)
        - frequency of threshold values: count and percentage
    """
    # Summary per CYP
    cyp_summary = thresholds_df.groupby("cyp")["threshold"].agg(
        mean_threshold="mean",
        std_threshold="std",
        min_threshold="min",
        max_threshold="max"
    ).round(4).reset_index()

    # Reorder rows according to CYP_ORDER
    cyp_summary['cyp'] = pd.Categorical(cyp_summary['cyp'], categories=CYP_ORDER, ordered=True)
    cyp_summary = cyp_summary.sort_values('cyp').reset_index(drop=True)

    # Frequency of thresholds
    freq = thresholds_df["threshold"].value_counts().reset_index()
    freq.columns = ["threshold", "count"]
    freq["percentage"] = (freq["count"] / freq["count"].sum() * 100).round(2)
    freq = freq.sort_values("threshold").reset_index(drop=True)

    return cyp_summary, freq


# ----------------------------------------------------------------------
# Baseline comparison (with corrected delta for hamming loss)
# ----------------------------------------------------------------------

def create_baseline_comparison(master_df: pd.DataFrame, baseline_metrics: Dict) -> pd.DataFrame:
    """
    Compare each supervised experiment to the majority baseline.

    For higher‑is‑better metrics (MCC, F1): delta = model - baseline.
    For lower‑is‑better metrics (hamming loss): delta = baseline - model.
    """
    if baseline_metrics is None:
        warnings.warn("Majority baseline not loaded; returning empty dataframe.")
        return pd.DataFrame()

    baseline_mcc = baseline_metrics.get("macro_mcc", 0.0)
    baseline_f1 = baseline_metrics.get("micro_f1", 0.0)
    baseline_hamming = baseline_metrics.get("hamming_loss", 0.0)

    records = []
    for _, row in master_df.iterrows():
        records.append({
            "model": row["model"],
            "representation": row["representation"],
            "macro_mcc": row["macro_mcc"],
            "baseline_macro_mcc": baseline_mcc,
            "delta_macro_mcc": row["macro_mcc"] - baseline_mcc,          # higher is better
            "micro_f1": row["micro_f1"],
            "baseline_micro_f1": baseline_f1,
            "delta_micro_f1": row["micro_f1"] - baseline_f1,            # higher is better
            "hamming_loss": row["hamming_loss"],
            "baseline_hamming_loss": baseline_hamming,
            "delta_hamming_loss": baseline_hamming - row["hamming_loss"] # lower is better → positive means improvement
        })

    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# Pivot tables (with fixed CYP order and added mean row/column)
# ----------------------------------------------------------------------

def create_pivot_tables(thresholds_df: pd.DataFrame, per_cyp_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create:
        - thresholds pivot: rows = CYP (ordered by CYP_ORDER), columns = (model, representation)
        - per‑CYP MCC pivot: rows = CYP (ordered by CYP_ORDER), columns = (model, representation)
          with an added 'mean' column (row average) and a 'mean' row (column average).
    """
    # Threshold pivot
    th_pivot = thresholds_df.pivot_table(
        index="cyp",
        columns=["model", "representation"],
        values="threshold"
    )
    # Reindex rows using fixed CYP_ORDER
    th_pivot = th_pivot.reindex(CYP_ORDER)

    # Per‑CYP MCC pivot - start without the mean column
    mcc_pivot = per_cyp_df.pivot_table(
        index="cyp",
        columns=["model", "representation"],
        values="mcc"
    )
    # Reindex rows using fixed CYP_ORDER
    mcc_pivot = mcc_pivot.reindex(CYP_ORDER)

    # Add a column with the mean MCC for each CYP (across all model‑representation pairs)
    mcc_pivot["mean"] = mcc_pivot.mean(axis=1).round(4)

    # Add a row with the mean MCC for each model‑representation (across all CYPs)
    # This row will also have a value in the 'mean' column (overall average)
    mean_row = mcc_pivot.mean(axis=0).round(4)
    mean_row.name = "mean"
    # Convert to DataFrame and concatenate (avoid deprecated .append)
    mean_row_df = pd.DataFrame(mean_row).T   # one row, columns same as mcc_pivot
    mcc_pivot = pd.concat([mcc_pivot, mean_row_df])

    return th_pivot, mcc_pivot


# ----------------------------------------------------------------------
# Export functions
# ----------------------------------------------------------------------

def export_tables(
    master_df: pd.DataFrame,
    cyp_summary: pd.DataFrame,
    th_freq: pd.DataFrame,
    summary_model: pd.DataFrame,
    summary_repr: pd.DataFrame,
    baseline_comp: pd.DataFrame,
    cv_df: pd.DataFrame,
    th_pivot: pd.DataFrame,
    mcc_pivot: pd.DataFrame
):
    """
    Export all dataframes to CSV files.

    Note: thresholds_df and per_cyp_df are not exported as separate files
    because their information is fully contained in the pivots.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    master_df.to_csv(OUTPUT_DIR / "master_results.csv", index=False)
    summary_model.to_csv(OUTPUT_DIR / "summary_by_model.csv", index=False)
    summary_repr.to_csv(OUTPUT_DIR / "summary_by_representation.csv", index=False)
    baseline_comp.to_csv(OUTPUT_DIR / "baseline_comparison.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / "all_cv_results.csv", index=False)
    th_pivot.to_csv(OUTPUT_DIR / "thresholds_pivot.csv")
    mcc_pivot.to_csv(OUTPUT_DIR / "per_cyp_mcc_pivot.csv")
    cyp_summary.to_csv(OUTPUT_DIR / "threshold_summary.csv", index=False)
    th_freq.to_csv(OUTPUT_DIR / "threshold_frequency.csv", index=False)

    print(f"All tables exported to {OUTPUT_DIR}/")


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def main():
    print("Collecting results...")

    # 1. Load majority baseline
    baseline_metrics = load_majority_baseline(MAJORITY_FILE)
    if baseline_metrics:
        print("Loaded majority baseline.")

    # 2. Collect all supervised experiments
    experiments, all_cv_df, thresholds_df = collect_all_experiments()
    print(f"Collected {len(experiments)} experiments")

    # 3. Build dataframes
    master_df = build_master_dataframe(experiments)
    per_cyp_df = build_per_cyp_dataframe(experiments)
    print(f"Collected {len(per_cyp_df)} per‑CYP results")

    if not all_cv_df.empty:
        print(f"Collected {len(all_cv_df)} CV runs")

    # 4. Summary tables
    summary_repr = summarise_by_representation(master_df)
    summary_model = summarise_by_model(master_df)

    # 5. Threshold analysis (now with ordered CYPs)
    cyp_summary, th_freq = analyse_thresholds(thresholds_df)

    # 6. Baseline comparison
    baseline_comp = create_baseline_comparison(master_df, baseline_metrics)

    # 7. Pivot tables (with ordered CYPs and mean row/column)
    th_pivot, mcc_pivot = create_pivot_tables(thresholds_df, per_cyp_df)

    # 8. Export
    export_tables(
        master_df=master_df,
        cyp_summary=cyp_summary,
        th_freq=th_freq,
        summary_model=summary_model,
        summary_repr=summary_repr,
        baseline_comp=baseline_comp,
        cv_df=all_cv_df,
        th_pivot=th_pivot,
        mcc_pivot=mcc_pivot
    )

    print("Done.")


if __name__ == "__main__":
    main()