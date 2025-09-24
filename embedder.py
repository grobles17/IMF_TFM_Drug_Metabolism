from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

def embed_inchi(InChI: str) -> NDArray[np.float64]:
    """Embeds a InChI string into a vector"""
    pass

def embed_smiles(smiles: str) -> NDArray[np.float64]:
    """Embeds a InChI string into a vector"""
    pass

# Create the generator once at import time
def get_morgan_generator(radius: int = 3, n_bits: int = 2048):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

# Default global generator (common case)
_default_gen = get_morgan_generator()

def smiles_to_morgan_fingerprint(smiles: str, generator = _default_gen) -> NDArray[np.int16] | None:
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
    np.ndarray of shape (fpSize,) and dtype int32
        Count fingerprint vector where each entry indicates the frequency
        of the corresponding environment. Returns None if the SMILES
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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    count_fp = generator.GetCountFingerprint(mol)
    return np.array(count_fp, dtype=np.int16)