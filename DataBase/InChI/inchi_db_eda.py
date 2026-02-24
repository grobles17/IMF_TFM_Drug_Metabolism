import pandas as pd
import os
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

script_dir = os.path.dirname(os.path.abspath(__file__))
inchi_file = os.path.join(script_dir, "inchi_output.txt")
db = pd.read_csv(inchi_file, header=None, names=["InChI"], sep="\t")

# --- 1. Load trained tokenizer ---
# (Adjust path to where you saved it)
tokenizer = ByteLevelBPETokenizer(
    vocab=os.path.join(script_dir, "inchi_tokenizer", "vocab.json"),
    merges=os.path.join(script_dir, "inchi_tokenizer", "merges.txt")
)
print("Loaded ByteLevelBPETokenizer")

# --- Function to tokenize all InChIs and save token IDs ---
def main():
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
        print("✓ Verification passed: saved tokens match fresh encoding.")

import numpy as np
import os

def compute_token_length_percentiles():
    tokens_file = os.path.join(script_dir, "inchi_tokens.txt")

    lengths = []
    with open(tokens_file, 'r') as f:
        print(f"Computing token length statistics from {tokens_file}...")
        for i, line in enumerate(f):
            token_count = len(line.split())
            lengths.append(token_count)

    # Convert to numpy array for percentile calculation
    lengths_np = np.array(lengths)
    percentiles = [50, 75, 90, 99, 99.9]
    p = np.percentile(lengths_np, percentiles)
    print(f"\n--- Token Length Statistics ---")
    print(f"Total sequences: {len(lengths)}")
    print(f"Mean length: {np.mean(lengths_np):.2f}")
    print(f"Median: {np.median(lengths_np)}")
    print(f"Max length: {np.max(lengths_np)}")
    print(f"Min length: {np.min(lengths_np)}")
    for perc, val in zip(percentiles, p):
        print(f"{perc}th percentile: {val}")

if __name__ == "__main__":
    #main() # Uncomment this line to run the tokenization when executing the script

    # --- 1. Max raw string length ---
    max_str_len = max(db["InChI"].apply(len))
    print(f"Longest InChI (characters): {max_str_len}")
    
    # --- 3. Find the InChI with maximum string length and tokenize it ---
    # Get the actual string
    longest_inchi = db.loc[db["InChI"].str.len().idxmax(), "InChI"]
    # Tokenize
    encoded = tokenizer.encode(longest_inchi)
    token_len = len(encoded.ids)
    print(f"Longest InChI tokenized length: {token_len} tokens")
    print(f"Sample tokens: {encoded.tokens[:15]}...")  # show first 15
    # --- Verification: compare a random line from saved file with fresh encoding ---
    #verify_saved_tokens_random()

    compute_token_length_percentiles()   