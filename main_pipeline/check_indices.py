"""
Check that all molecular representation TSV files share identical indices.
Run from the same directory as xgboost_base_model.py.
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPR_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "representations"))

REPR_FILES = [
    "morgan_output_representation.tsv",
    "MolE_output_representation.tsv",
    "chemberta_output_representation.tsv",
    "inchi_output_representation.tsv",
]

def load_index(filepath):
    """Load only the index column of a TSV (fast, no feature columns read)."""
    df = pd.read_csv(filepath, sep="\t", index_col=0, usecols=[0])
    return df.index

def main():
    print("=" * 60)
    print("Index consistency check across representation files")
    print("=" * 60)

    indices = {}
    missing = []

    for fname in REPR_FILES:
        path = os.path.join(REPR_DIR, fname)
        if not os.path.exists(path):
            print(f"  [MISSING]  {fname}")
            missing.append(fname)
            continue
        idx = load_index(path)
        indices[fname] = idx
        print(f"  [LOADED]   {fname}  ({len(idx)} molecules)")

    if missing:
        print(f"\nCannot proceed: {len(missing)} file(s) not found.")
        return

    files = list(indices.keys())
    reference_file = files[0]
    reference_idx  = indices[reference_file]
    reference_set  = set(reference_idx)

    print(f"\nReference: {reference_file}")
    print("-" * 60)

    all_match = True

    for fname in files[1:]:
        idx      = indices[fname]
        other_set = set(idx)

        only_in_reference = reference_set - other_set
        only_in_other     = other_set - reference_set

        if not only_in_reference and not only_in_other:
            # Sets match — also check order
            if list(reference_idx) == list(idx):
                print(f"  [OK]  {fname} — identical indices and order")
            else:
                all_match = False
                print(f"  [ORDER MISMATCH]  {fname} — same IDs but different row order")
                # Show first differing position
                for pos, (a, b) in enumerate(zip(reference_idx, idx)):
                    if a != b:
                        print(f"    First difference at row {pos}: '{a}' vs '{b}'")
                        break
        else:
            all_match = False
            print(f"  [MISMATCH]  {fname}")
            if only_in_reference:
                sample = sorted(only_in_reference)[:10]
                print(f"    In reference but NOT in this file ({len(only_in_reference)} IDs):")
                for mid in sample:
                    print(f"      {mid}")
                if len(only_in_reference) > 10:
                    print(f"      ... and {len(only_in_reference) - 10} more")
            if only_in_other:
                sample = sorted(only_in_other)[:10]
                print(f"    In this file but NOT in reference ({len(only_in_other)} IDs):")
                for mid in sample:
                    print(f"      {mid}")
                if len(only_in_other) > 10:
                    print(f"      ... and {len(only_in_other) - 10} more")

    print("-" * 60)
    if all_match:
        print("SUCCESS: all representation files share identical indices and row order.")
    else:
        print("FAILED: mismatches found — see details above.")
    print("=" * 60)

if __name__ == "__main__":
    main()