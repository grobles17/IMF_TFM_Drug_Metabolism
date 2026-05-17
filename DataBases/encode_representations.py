from pyexpat import model

import numpy as np
import pandas as pd

from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import torch
import numpy as np
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer

# MolE representations were obtained following the instructions in the original repository:
# https://github.com/rolayoalarcon/MolE?tab=readme-ov-file
# Using a dedicated conda environment with the specified dependencies, 
# and running the provided scripts to generate the representations for our dataset. 
# The resulting TSV file was then read into a DataFrame for analysis.

import torch
from transformers import AutoTokenizer, AutoModel

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "representations"))

### Morgan fingerprint generation using RDKit's rdFingerprintGenerator with count-based encoding. ###
# Create the generator once at import time
def get_morgan_generator(radius: int = 3, n_bits: int = 2048):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

# Default global generator (common case)
_default_gen = get_morgan_generator()

def smiles_to_morgan_fingerprint(smiles: str, generator = _default_gen) -> NDArray[np.int16]:
    """
    Generate a count-based Morgan fingerprint from a SMILES string.
    
    This function encodes the molecular structure into a fixed-length
    integer vector using the Morgan fingerprint algorithm.
    It returns a count-based fingerprint, retaining the number of 
    occurrences of each local chemical environment

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule to be embedded.
    generator : rdFingerprintGenerator.FingerprintGenerator, optional
        Preconfigured fingerprint generator. By default, a Morgan
        generator with radius=3 and fpSize=2048 is used.
    
    Returns
    -------
    np.ndarray of shape (fpSize,) and dtype int16
        Count fingerprint vector where each entry indicates the frequency
        of the corresponding environment. Returns zero array if the SMILES
        string cannot be parsed.
    
    Notes
    -----
    - `int16` is chosen to avoid overflow: it can represent counts up to
      ~32k, which is far beyond the expected frequency of any
      substructure in drug-like molecules.
    - Compared to binary fingerprints, count fingerprints are generally
      more informative for predictive modeling of metabolic liability,
      as they preserve multiplicities of moieties that may influence
      enzyme-substrate interactions.
    """
    N_BITS = generator.GetOptions().fpSize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(N_BITS, dtype=np.int16) 
    count_fp = generator.GetCountFingerprint(mol)
    # Proper conversion from sparse vector to dense array
    fp_array = np.zeros(N_BITS, dtype=np.int16)
    
    # Fill in the non-zero elements
    for idx, count_val in count_fp.GetNonzeroElements().items():
        fp_array[idx] = count_val
    return fp_array

### ChemBERTa embedding generation using Hugging Face Transformers. ###
# Load the pre-trained ChemBERTa model and tokenizer once at import time
_CHEMBERTA_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

_tokenizer = AutoTokenizer.from_pretrained(_CHEMBERTA_MODEL_NAME)
_chemberta_model = AutoModel.from_pretrained(_CHEMBERTA_MODEL_NAME, use_safetensors=True)
_chemberta_model.eval()  # disable dropout

def chemberta_embedder(smiles: str) -> NDArray[np.float32]:
    """
    Generate a transformer-based molecular embedding from a SMILES string
    using a pretrained ChemBERTa model.

    This function encodes the molecular structure into a fixed-length
    continuous vector by tokenizing the SMILES string and forwarding it
    through a pretrained transformer encoder. A mean pooling operation
    over token embeddings (excluding padding tokens) is applied to obtain
    a single molecular representation.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule to be embedded.

    Returns
    -------
    np.ndarray of shape (hidden_size,) and dtype float64
        Continuous embedding vector representing the molecule in latent
        space.

    Notes
    -----
    - A maximum token length of 1024 is used to avoid truncation of large
    molecules.
    - The output is cast to `float64` for compatibility with MolE
    embeddings, ensuring consistent numeric types across learned
    representations in the project.
    """
    inputs = _tokenizer(
        smiles,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024) #vancomycin's SMILES has more than 700 tokens, 
        #so we set a higher limit to avoid truncation for large molecules

    with torch.no_grad():
        outputs = _chemberta_model(**inputs)

    # Mean pooling over token embeddings (excluding padding)
    last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    masked_hidden = last_hidden * attention_mask
    sum_hidden = masked_hidden.sum(dim=1)
    valid_tokens = attention_mask.sum(dim=1)

    embedding = sum_hidden / valid_tokens

    return embedding.squeeze(0).cpu().numpy().astype(np.float64) #float 64 is used for compatibility with MolE embeddings, which are also float64.

# InChI Embedder (Author-trained MLM)
_INCHI_MODEL_PATH = "./inchi_embedder_final"

_inchi_tokenizer = AutoTokenizer.from_pretrained(_INCHI_MODEL_PATH)
_inchi_model = AutoModel.from_pretrained(
    _INCHI_MODEL_PATH,
    add_pooling_layer=False  # remove unused pooler
)
_inchi_model.eval()  # disable dropout

def inchi_embedder(inchi: str) -> NDArray[np.float64]:
    """
    Generate a transformer-based molecular embedding from an InChI string
    using the user-trained MLM.

    This function encodes the molecular structure into a fixed-length
    continuous vector by tokenizing the InChI string and forwarding it
    through the pretrained transformer encoder. A mean pooling operation
    over token embeddings (excluding padding tokens) is applied to obtain
    a single molecular representation.

    Parameters
    ----------
    inchi : str
        InChI representation of the molecule to be embedded.

    Returns
    -------
    np.ndarray of shape (hidden_size,) and dtype float64
        Continuous embedding vector representing the molecule in latent
        space.

    Notes
    -----
    - A maximum token length of 512 is used, matching the configuration
      employed during MLM pretraining.
    - Sequences longer than 512 tokens are truncated to ensure
      architectural consistency with the trained model
      (max_position_embeddings = 514).
    - The output is cast to float64 for comparability with MolE and
      ChemBERTa embeddings, ensuring consistent numeric types across
      learned representations in the project.
    """

    inputs = _inchi_tokenizer(
        inchi,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512  # MUST match training configuration
    )

    with torch.no_grad():
        outputs = _inchi_model(**inputs)

    # Mean pooling over token embeddings (excluding padding)
    last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    masked_hidden = last_hidden * attention_mask
    sum_hidden = masked_hidden.sum(dim=1)
    valid_tokens = attention_mask.sum(dim=1)

    embedding = sum_hidden / valid_tokens

    return embedding.squeeze(0).cpu().numpy().astype(np.float64)

def featurize_smiles(structure: str, method: str = "morgan") -> NDArray[np.int16 | np.float64]:
    """
    Featurize a SMILES string into a numerical vector using the specified method.

    This function provides a unified interface for molecular representation,
    allowing interchangeable use of handcrafted fingerprints and learned
    embeddings within the modeling pipeline.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule to be embedded.
    method : str, default="morgan"
        Featurization strategy to apply. Supported options are:
        - "morgan": count-based Morgan fingerprint (int16).
        - "chemberta": transformer-based embedding (float64).
        - "inchi": user-trained transformer-based embedding from InChI (float64).

    Returns
    -------
    np.ndarray
        Numerical feature vector representing the molecule. The dtype
        depends on the selected method.

    Raises
    ------
    ValueError
        If an unsupported featurization method is specified.
    """
    if method == "morgan":
        return smiles_to_morgan_fingerprint(structure).astype(np.int16)
    elif method == "chemberta":
        return chemberta_embedder(structure).astype(np.float64)
    elif method == "inchi":
        return inchi_embedder(structure).astype(np.float64)
    else:
        raise ValueError(f"Unknown featurization method: {method}")

    
if __name__ == "__main__":
    db = pd.read_csv(os.path.join(SCRIPT_DIR, "DrugBank_curated_df.csv"))

    ids = []
    fingerprints = []
    chemberta_embeddings = []
    inchis = []

    for compound_id, smiles, inchi_str in zip(
    db["DrugBank ID"],
    db["SMILES"],
    db["InChI"]):
        fp = featurize_smiles(smiles, method="morgan")
        emb = featurize_smiles(smiles, method="chemberta")
        inchi = featurize_smiles(inchi_str, method="inchi")
        ids.append(compound_id)
        fingerprints.append(fp)
        chemberta_embeddings.append(emb)
        inchis.append(inchi)

    # Convert to 2D numpy array
    fingerprint_matrix = np.vstack(fingerprints)
    chemberta_matrix = np.vstack(chemberta_embeddings)
    inchi_matrix = np.vstack(inchis)

    # Create DataFrame with explicit column names
    morgan_df = pd.DataFrame(
        fingerprint_matrix,
        index=ids,
        columns=[str(i) for i in range(fingerprint_matrix.shape[1])]
    )
    chemberta_df = pd.DataFrame(
        chemberta_matrix,
        index=ids,
        columns=[str(i) for i in range(chemberta_matrix.shape[1])]
    )
    inchi_df = pd.DataFrame(
        inchi_matrix,
        index=ids,
        columns=[str(i) for i in range(inchi_matrix.shape[1])]
    )
    # Save as TSV with ID as index
    morgan_df.to_csv(os.path.join(SCRIPT_DIR,
        "morgan_output_representation.tsv"),
        sep="\t", # Separatar is tab for consistency with previous files
        index_label="DrugBank ID"
    )

    chemberta_df.to_csv(os.path.join(SCRIPT_DIR,
        "chemberta_output_representation.tsv"),
        sep="\t",
        index_label="DrugBank ID"
    )

    inchi_df.to_csv(os.path.join(SCRIPT_DIR,
        "inchi_output_representation.tsv"),
        sep="\t",
        index_label="DrugBank ID"
    )

    mole_df = pd.read_csv(os.path.join(OUTPUT_DIR, "MolE_output_representation.tsv"), 
                          sep="\t", 
                          index_col=0)
    print("Shape of the MolE representation DataFrame:")
    print(mole_df.shape)
    print("Shape of the Morgan fingerprint DataFrame:")
    print(morgan_df.shape)
    print(round(morgan_df.memory_usage(deep=True).sum() / 1024**2, 2), "MB")
    print("Shape of the ChemBERTa embedding DataFrame:")
    print(chemberta_df.shape)
    print("Shape of the InChI embedding DataFrame:")
    print(inchi_df.shape)