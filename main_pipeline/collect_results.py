"""
Loads and organises all model/representation results from the
IMF_TFM_Drug_Metabolism project into analysis‑ready tables.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------

# Root directory containing the results
RESULTS_ROOT = Path("./main_pipeline/results")

# Models to process (majority_baseline is ignored)
MODELS = ["logistic_regression", "random_forest", "xgboost_results"]

# Representations (the folders inside each model directory)
REPRESENTATIONS = ["MolE", "chemberta", "inchi", "morgan"]

# Suffix used in filenames (matches the representation name)
# e.g. best_params_chemberta.json, best_thresholds_per_cyp_chemberta.json
REPR_SUFFIX = {
    "MolE": "mole",        # adjust if actual files use "mole" or "MolE"
    "chemberta": "chemberta",
    "inchi": "inchi",
    "morgan": "morgan"
}

# ----------------------------------------------------------------------
# 2. Helper functions
# ----------------------------------------------------------------------

def find_files(model: str, repr_name: str) -> Dict[str, Path]:
    """
    Locate the four result files for a given (model, representation).

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

    # Check existence
    for key, path in files.items():
        if not path.exists():
            print(f"Warning: {path} not found")
            files[key] = None

    return files


def load_thresholds(file_path: Path) -> pd.Series:
    """Load thresholds JSON and return as a Series (CYP -> threshold)."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    # data is expected to be a dict like {"CYP1A2": 0.45, ...}
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


# ----------------------------------------------------------------------
# 3. Main collection routine
# ----------------------------------------------------------------------

def collect_all_results() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Iterate over all models and representations, load the data,
    and return three tidy DataFrames:

    1. thresholds_df  : index = (model, representation, CYP)
    2. params_df      : index = (model, representation) with parameter columns
    3. metrics_df     : index = (model, representation) with metric columns
    """
    thresholds_records = []
    params_records = []
    metrics_records = []

    for model in MODELS:
        for repr_name in REPRESENTATIONS:
            files = find_files(model, repr_name)

            # --- Thresholds ---
            if files["thresholds"] is not None:
                th_series = load_thresholds(files["thresholds"])
                for cyp, th in th_series.items():
                    thresholds_records.append({
                        "model": model,
                        "representation": repr_name,
                        "cyp": cyp,
                        "threshold": th
                    })

            # --- Best parameters ---
            if files["params"] is not None:
                params = load_params(files["params"])
                # Flatten the dict (assumes one level of nesting)
                flat_params = {"model": model, "representation": repr_name}
                for k, v in params.items():
                    # If v is a dict, you might want to flatten further
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            flat_params[f"{k}_{sub_k}"] = sub_v
                    else:
                        flat_params[k] = v
                params_records.append(flat_params)

            # --- Test metrics ---
            if files["metrics"] is not None:
                metrics = load_metrics(files["metrics"])
                flat_metrics = {"model": model, "representation": repr_name}
                for k, v in metrics.items():
                    flat_metrics[k] = v
                metrics_records.append(flat_metrics)

            # --- CV results (optional) ---
            # If you want to store cv_results, you can load them separately
            # and merge later. For now we skip them to keep the tables clean.

    # Build DataFrames
    thresholds_df = pd.DataFrame(thresholds_records)
    params_df = pd.DataFrame(params_records)
    metrics_df = pd.DataFrame(metrics_records)

    return thresholds_df, params_df, metrics_df


# ----------------------------------------------------------------------
# 4. Export and summary functions
# ----------------------------------------------------------------------

def export_tables(thresholds_df: pd.DataFrame,
                  params_df: pd.DataFrame,
                  metrics_df: pd.DataFrame,
                  output_dir: str = "./collected_results"):
    """Save the three DataFrames as CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    thresholds_df.to_csv(os.path.join(output_dir, "thresholds_per_cyp.csv"), index=False)
    params_df.to_csv(os.path.join(output_dir, "best_params.csv"), index=False)
    metrics_df.to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)

    print(f"Tables saved to {output_dir}/")


def create_pivot_tables(thresholds_df: pd.DataFrame,
                        metrics_df: pd.DataFrame):
    """
    Create pivot tables that make comparison easier.

    - thresholds_pivot: rows = CYP, columns = (model, representation)
    - metrics_pivot: rows = (model, representation), columns = metric
    """
    # Pivot thresholds: CYP as rows, (model, representation) as columns
    th_pivot = thresholds_df.pivot_table(
        index="cyp",
        columns=["model", "representation"],
        values="threshold"
    )

    # Pivot metrics: (model, representation) as rows, metrics as columns
    metrics_pivot = metrics_df.set_index(["model", "representation"])

    return th_pivot, metrics_pivot


# ----------------------------------------------------------------------
# 5. Example analysis helpers
# ----------------------------------------------------------------------

def compare_representations(metrics_df: pd.DataFrame, metric: str = "mcc"):
    """
    For a given metric (e.g. 'mcc'), show average performance per representation
    across all models and CYPs.
    """
    # Average over models and CYPs (if metrics are per‑CYP, you'd need to aggregate)
    # Here we assume metrics_df has one row per (model, representation)
    return metrics_df.groupby("representation")[metric].mean().sort_values(ascending=False)


def compare_models(metrics_df: pd.DataFrame, metric: str = "mcc"):
    """Show average performance per model for a given metric."""
    return metrics_df.groupby("model")[metric].mean().sort_values(ascending=False)


# ----------------------------------------------------------------------
# 6. Main execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Collect all data
    thresholds_df, params_df, metrics_df = collect_all_results()

    # 2. Export raw tables
    export_tables(thresholds_df, params_df, metrics_df)

    # 3. Create pivot tables for easier visual comparison
    th_pivot, metrics_pivot = create_pivot_tables(thresholds_df, metrics_df)

    # 4. Print some quick summaries
    print("\n=== Thresholds (first 5 rows) ===")
    print(th_pivot.head())

    print("\n=== Metrics (first 5 rows) ===")
    print(metrics_pivot.head())

    print("\n=== Average MCC per representation ===")
    print(compare_representations(metrics_df, "mcc"))

    print("\n=== Average MCC per model ===")
    print(compare_models(metrics_df, "mcc"))

    # 5. (Optional) Save pivots as well
    th_pivot.to_csv("./collected_results/thresholds_pivot.csv")
    metrics_pivot.to_csv("./collected_results/metrics_pivot.csv")
