from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem
import numpy as np


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
    N_BITS = 2048 #Modify length if Generator's n_bits != 2048
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

def featurize_smiles(smiles: str, method: str = "morgan") -> NDArray[np.float64]:
    """Featurizes a SMILES string into a vector using the specified method. 
    
    Defaults to Morgan fingerprints if no method is specified. 
    Returns a float64 array for compatibility.
    
    Raises ValueError if an unknown method is specified."""
    if method == "morgan":
        return smiles_to_morgan_fingerprint(smiles).astype(np.float64)
    else:
        raise ValueError(f"Unknown featurization method: {method}")