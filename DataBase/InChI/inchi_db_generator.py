import time
import pandas as pd
from rdkit import Chem


def smiles_to_inchi(smiles: str) -> str | None:
    """
    Convert a SMILES string to an InChI string.
    Returns the InChI on success, None on failure (invalid SMILES or RDKit error).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi = Chem.MolToInchi(mol)
        return inchi if inchi is not None else None
    except Exception:
        # Catch any unexpected RDKit exceptions gracefully
        return None


def main_inchi_db():
    # ========== CONFIGURATION ==========
    input_file = "smiles_chembl.smi"
    output_file = "inchi_output.txt" 
    sep = "\t"                     
    names = ["SMILES", "ChEMBL_ID"] # input has no header
    chunk_size = 10000              # process in chunks to avoid huge memory use

    print(f"Reading SMILES from {input_file} ...")
    start_time = time.time()

    total_rows = 0
    success_count = 0
    failure_count = 0

    # Open output file for writing incrementally
    with open(output_file, "w", encoding="utf-8") as out_f:
        # Process the large file in chunks
        reader = pd.read_csv(
            input_file,
            sep=sep,
            names=names,
            chunksize=chunk_size,
            dtype=str,           # read everything as string to avoid type issues
            keep_default_na=False # keep empty strings as is
        )

        for chunk in reader:
            for smi in chunk["SMILES"]:
                total_rows += 1
                inchi = smiles_to_inchi(smi)
                if inchi:
                    out_f.write(inchi + "\n")
                    success_count += 1
                else:
                    failure_count += 1

            # Optional: flush output occasionally (not required, but safe)
            out_f.flush()

    end_time = time.time()
    elapsed = end_time - start_time

    # ========== STATISTICS ==========
    print("\n" + "="*50)
    print("CONVERSION COMPLETE")
    print("="*50)
    print(f"Total SMILES processed:    {total_rows:,}")
    print(f"Successful conversions:    {success_count:,}")
    print(f"Failed conversions:        {failure_count:,}")
    if total_rows > 0:
        success_rate = (success_count / total_rows) * 100
        print(f"Success rate:              {success_rate:.2f}%")
    print(f"Total time:                {elapsed:.2f} seconds")
    if total_rows > 0:
        avg_per_smiles = elapsed / total_rows
        print(f"Average time per SMILES:   {avg_per_smiles*1000:.2f} ms")
    if success_count > 0:
        avg_per_success = elapsed / success_count
        print(f"Average time per success:  {avg_per_success*1000:.2f} ms")
    print(f"Output saved to:           {output_file}")
    print("="*50)


if __name__ == "__main__":
    # main_inchi_db()  # Uncomment to run the conversion
    inchi_db = pd.read_csv("./DataBase/InChI/inchi_output.txt", header=None, names=["InChI"], sep="\t")
    print(inchi_db.head())
