import pandas as pd
import os
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

script_dir = os.path.dirname(os.path.abspath(__file__))
inchi_file = os.path.join(script_dir, "inchi_output.txt")
db = pd.read_csv(inchi_file, header=None, names=["InChI"], sep="\t")
db_smiles = pd.read_csv(os.path.join(script_dir, "smiles_chembl.smi"), header=None, names = ["SMILES", "ChEMBL_ID"], sep="\t")
tokens_file = os.path.join(script_dir, "inchi_tokens.txt")
# --- Load trained tokenizer ---
tokenizer = ByteLevelBPETokenizer(
    vocab=os.path.join(script_dir, "inchi_tokenizer", "vocab.json"),
    merges=os.path.join(script_dir, "inchi_tokenizer", "merges.txt")
)
print("Loaded ByteLevelBPETokenizer")

# --- Function to tokenize all InChIs and save token IDs ---
def main():
    """Tokenize all InChI strings in the global database and save the token IDs.

    The function encodes each InChI using the pre-loaded tokenizer and writes the resulting
    token IDs to `inchi_tokens.txt` (one line per InChI, space-separated tokens).
    Progress is printed every 100,000 processed entries.
    """
    output_file = os.path.join(script_dir, "inchi_tokens.txt")
    total = len(db)
    print(f"Starting tokenization of {total} InChIs...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, inchi in enumerate(db["InChI"]):
            encoded = tokenizer.encode(inchi)
            token_ids = encoded.ids
            f_out.write(" ".join(map(str, token_ids)) + "\n")
            if (i + 1) % 100000 == 0:
                print(f"Processed {i+1}/{total} InChIs")
    print(f"Tokenization complete. Output saved to {output_file}")

import random
# --- Verification: compare a random line from saved file with fresh encoding ---
def verify_saved_tokens_random():
    """Verify that saved token IDs match fresh encoding for a random InChI.

    This function reads the cached token IDs from `inchi_tokens.txt`, selects a
    random line, and compares the saved IDs against a freshly encoded version of
    the original InChI string. It also prints the decoded InChI for visual
    confirmation. The check is deterministic and serves as a sanity test for
    tokenization consistency.

    Assumes:
        - `db` is a pandas DataFrame with a column 'InChI'.
        - `tokenizer` is a loaded ByteLevelBPETokenizer instance.
        - `script_dir` points to the directory containing `inchi_tokens.txt`.
    """
    tokens_file = os.path.join(script_dir, "inchi_tokens.txt")
    if not os.path.exists(tokens_file):
        print(f"Error: {tokens_file} not found.")
        return

    # Read all lines to count them and pick a random line
    with open(tokens_file, 'r') as f:
        lines = f.readlines()
    total_lines = len(lines)

    # Pick a random line number (1‑based)
    line_number = random.randint(1, total_lines)
    saved_ids = list(map(int, lines[line_number-1].strip().split()))

    # Get the original InChI from the dataframe (0‑based index)
    original_inchi = db.iloc[line_number-1]["InChI"]

    # Freshly encode the original InChI
    fresh_encoded = tokenizer.encode(original_inchi)
    fresh_ids = fresh_encoded.ids

    # Decode the saved IDs back to a string
    decoded_str = tokenizer.decode(saved_ids)

    print(f"=== Verification for random line {line_number} (of {total_lines}) ===")
    print(f"Original InChI        : {original_inchi}")
    print(f"Decoded from saved IDs: {decoded_str}")
    print(f"Token IDs match       : {saved_ids == fresh_ids}")

    if saved_ids != fresh_ids:
        print("\nMismatch details:")
        print(f"Saved IDs : {saved_ids}")
        print(f"Fresh IDs : {fresh_ids}")
        print(f"Saved tokens : {tokenizer.decode(saved_ids, skip_special_tokens=False)}")
        print(f"Fresh tokens : {tokenizer.decode(fresh_ids, skip_special_tokens=False)}")
    else:
        print("Verification passed: saved tokens match fresh encoding.")

import numpy as np
import os

import numpy as np

# --- Compute InChI length percentiles ---
def compute_inchi_length_percentiles():
    """Compute character‑length percentiles for the InChI database file.

    Reads the InChI strings from `inchi_output.txt`, calculates the character 
    length of each line, and prints summary statistics including mean, median, 
    min, max, and specified percentiles (1, 50, 75, 90, 99, 99.9).
    """
    inchi_file = os.path.join(script_dir, "inchi_output.txt")

    lengths = []
    with open(inchi_file, 'r') as f:
        for i, line in enumerate(f):
            token_count = len(line.rstrip('\n'))
            lengths.append(token_count)

    # Convert to numpy array for percentile calculation
    lengths_np = np.array(lengths)
    percentiles = [1, 50, 75, 90, 99, 99.9]
    p = np.percentile(lengths_np, percentiles)
    print(f"\n--- InChI Length Statistics ---")
    print(f"Total sequences: {len(lengths)}")
    print(f"Mean length: {np.mean(lengths_np):.2f}")
    print(f"Median: {np.median(lengths_np)}")
    print(f"Max length: {np.max(lengths_np)}")
    print(f"Min length: {np.min(lengths_np)}")
    for perc, val in zip(percentiles, p):
        print(f"{perc}th percentile: {val}")

def analyze_inchi_structure(inchi):
    """Extract structural metrics from an InChI string.

    Parameters
    ----------
    inchi : str
        The InChI string to analyze.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'length' : int, total number of characters in the InChI string.
        - 'num_layers' : int, number of layers in the InChI (count of '/' separators).
        - 'has_stereo' : int, 1 if any stereochemistry layer marker ('/t', '/m', '/b', '/s')
          is present in the string, otherwise 0.
    """
    return {
        "length": len(inchi),
        "num_layers": inchi.count("/"),
        "has_stereo": int("/t" in inchi or "/m" in inchi or "/b" in inchi or "/s" in inchi),
    }

def analyze_duplicates(df, df_smiles):
    """Print duplicate and multiplicity statistics for InChI and SMILES datasets.

    This function computes the number of duplicate entries (rows after the first
    occurrence) in the InChI column of `df` and the SMILES column of `df_smiles`.
    It also reports the maximum number of occurrences of any single InChI or SMILES
    string. Results are printed to the console.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least an 'InChI' column.
    df_smiles : pandas.DataFrame
        DataFrame containing at least a 'SMILES' column. Must have the same
        number of rows as `df` (alignment by row index is assumed).

    Returns
    -------
    None
        The function prints the duplicate analysis and multiplicity information
        directly; nothing is returned.

    Notes
    -----
    The duplicate count uses `pandas.DataFrame.duplicated(subset=...)` which
    marks all rows after the first occurrence of a duplicated value as True.
    The multiplicity is the maximum frequency of any single string value.
    """
    dup_inchi = df.duplicated(subset="InChI").sum()
    dup_smiles = df_smiles.duplicated(subset="SMILES").sum()

    total = len(df)
    print(f"\n--- Duplicate Analysis ---")
    print(f"InChI duplicates: {dup_inchi} ({dup_inchi/total:.2%})")
    print(f"SMILES duplicates: {dup_smiles} ({dup_smiles/total:.2%})")

    unique_ratio = (total - dup_inchi) / total
    print(f"Unique molecule ratio: {unique_ratio:.4f}")
    print(f"Unique molecules: {total - dup_inchi:,} out of {total:,}")

    print("\n--- Multiplicity ---")
    print(f"Max duplicates (InChI): {df['InChI'].value_counts().max()}")
    print(f"Max duplicates (SMILES): {df_smiles['SMILES'].value_counts().max()}")


def compute_token_length_percentiles():
    """Compute token‑length percentiles from the tokenized InChI file.
    
    Same output as `compute_inchi_length_percentiles()`, but operates on 
    the tokenized version of the InChI data."""
    tokens_file = os.path.join(script_dir, "inchi_tokens.txt")

    lengths = []
    with open(tokens_file, 'r') as f:
        for i, line in enumerate(f):
            token_count = len(line.split())
            lengths.append(token_count)

    # Convert to numpy array for percentile calculation
    lengths_np = np.array(lengths)
    percentiles = [1, 50, 75, 90, 99, 99.9]
    p = np.percentile(lengths_np, percentiles)
    print(f"\n--- Token Length Statistics ---")
    print(f"Total sequences: {len(lengths)}")
    print(f"Mean length: {np.mean(lengths_np):.2f}")
    print(f"Median: {np.median(lengths_np)}")
    print(f"Max length: {np.max(lengths_np)}")
    print(f"Min length: {np.min(lengths_np)}")
    for perc, val in zip(percentiles, p):
        print(f"{perc}th percentile: {val}")

from collections import Counter

def token_frequency(inchi_tokens_file):
    counter = Counter()
    with open(inchi_tokens_file) as f: 
        for line in f:
            counter.update(line.split())
    
    def decode_token(t):
        return tokenizer.decode([int(t)])
    
    top = counter.most_common(5)
    bottom = counter.most_common()[-5:]
    print("\n--- Token Frequency Analysis ---")
    print("Top 5 tokens:")
    for t, c in top:
        print(f"'{decode_token(t)}' : {c}")

    print("\nBottom 5 tokens:")
    for t, c in bottom:
        print(f"'{decode_token(t)}' : {c}")
    rare_tokens = [t for t, c in counter.items() if c < 5]
    print(f"Number of rare tokens (count < 5): {len(rare_tokens)}")

if __name__ == "__main__":
    #main() # Uncomment this line to run the tokenization when first executing the script
    
    compute_inchi_length_percentiles()
    ### GRAPH
    total = len(db)
    num_layers = 0
    has_stereo_count = 0
    for i in db["InChI"]:
        struct = analyze_inchi_structure(i)
        num_layers += struct["num_layers"]
        has_stereo_count += struct["has_stereo"]

    print(f"\n--- InChI Structural Analysis ---")
    print(f"Average number of layers: {num_layers / total:.2f}")
    print(f"Percentage of InChIs with stereochemistry: {has_stereo_count / total * 100:.2f}%")
    print(f"Total stereochemistry entries: {has_stereo_count}")

    unique_chars = set("".join(db["InChI"]))
    print(f"\n--- Character Analysis ---")
    print(f"Number of unique characters: {len(unique_chars)}")
    
    analyze_duplicates(db, db_smiles)

    # Find SMILES with maximum duplication
    dup_counts = db_smiles["SMILES"].value_counts()
    top_n = 5
    print(f"\nTop {top_n} duplicated SMILES:")
    for smi, count in dup_counts.head(top_n).items():
        print(f"{count}: {smi}")

    # --- Count total tokens ---
    total_tokens = 0
    with open(tokens_file, "r") as f:
        for line in f:
            total_tokens += len(line.split())

    # --- Count total characters ---
    total_chars = 0
    with open(inchi_file, "r") as f:
        for line in f:
            total_chars += len(line.strip())

    # --- Compute averages ---
    num_sequences = sum(1 for _ in open(inchi_file))
    avg_chars = total_chars / num_sequences
    avg_tokens = total_tokens / num_sequences

    # --- Compute ratio ---
    chars_per_token = total_chars / total_tokens

    print("\n--- Tokenization Summary ---")
    print(f"Average characters per sequence: {avg_chars:.3f}")
    print(f"Average tokens per sequence: {avg_tokens:.3f}")
    print(f"Characters per token: {chars_per_token:.3f}")

    compute_token_length_percentiles()

    # --- Token frequency analysis ---
    token_frequency(tokens_file)
 
    # --- Verification: compare a random line from saved file with fresh encoding ---
    #verify_saved_tokens_random()