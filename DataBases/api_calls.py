from chemspipy import ChemSpider
from pandas import DataFrame
from typing import Any, Optional

from rdkit import Chem

from dataclasses import dataclass

# Environment-based API key loading
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

api_key = os.environ.get("CHEMSPI_API_KEY")
if not api_key:
    raise RuntimeError("CHEMSPI_API_KEY environment variable not set")

_cs_default = ChemSpider(api_key)  # ChemSpider API

@dataclass
class CompoundRecord:
    drugbank_id: str
    internal_code: str
    cyps: str
    query_name: str
    common_name: str | None
    smiles: str | None


def get_SMILES_chemspipy(missing_structures: DataFrame, 
                         cs_client = _cs_default)-> list[CompoundRecord]:
    """
    Retrieve SMILES strings for compounds using the ChemSpiPy API.

    This function takes a DataFrame of compounds with missing structural information 
    and queries ChemSpider (via the ChemSpiPy client object `cs`) for each compound name 
    provided. For each match found, it fetches detailed compound information, extracts 
    the canonical SMILES string, and assembles a dataclass containing key metadata fields 
    alongside the retrieved SMILES. These dataclass defined as CompoundRecord are collected 
    into a list, which can be unpacked conveniently.

    Parameters
    ----------
    missing_structures : pandas.DataFrame
        DataFrame containing compounds requiring SMILES retrieval. 
        Must include columns:
        - "DrugBank ID" : DrugBank identifier
        - "Name"        : Compound name for ChemSpider search
        - "CYPs"        : Associated CYP enzymes
    
    cs_client : ChemSpider Client
        The ChemSpiPy client to use. Uses `_cs_default` if not passed.
    
    Returns
    -------
    List[CompoundRecord]
        A list of CompoundRecord entries, each containing:
        (DrugBank ID, generated internal code, CYPs, compound name, ChemSpider common name, SMILES string)
        Internal code uses format "DBX0{row_idx}{match_count}".
    """
    match_count = 0
    compound_records: list[CompoundRecord] = []
    for row_idx, drug in missing_structures.iterrows():
        try:
            results = cs_client.search(drug["Name"])
        except Exception as e:
            print(f"ChemSpider search for {drug['Name']} failed: {e}")
            continue
        for result in results:
            try:
                match_count += 1
                compound = cs_client.get_compound_info(result.record_id)
                record = CompoundRecord(
                    drugbank_id=drug["DrugBank ID"],
                    internal_code=f"DBX0{row_idx}{match_count}",
                    cyps=drug["CYPs"],
                    query_name=drug["Name"],
                    common_name=compound.get("commonName"),
                    smiles=compound.get("smiles"),
                )
                compound_records.append(record)
            except Exception:
                #Skip compounds with incomplete info
                print(f"Failed to retrieve details for record ID {result.record_id} ({drug['Name']})")
                continue
    return compound_records


def smiles_to_inchi(smiles: str) -> Optional[str]:
    """
    Convert a SMILES string into its corresponding InChI representation.

    Parameters
    ----------
    smile : str
        A valid SMILES string representing the molecule.

    Returns
    -------
    inchi : str or None
        The InChI string corresponding to the input molecule,
        or None if the SMILES string cannot be parsed.

    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = str(Chem.MolToInchi(mol))
    return inchi

if __name__ == "__main__":
    print(smiles_to_inchi("CCOCO"))   # Should print a valid InChI string
    print(smiles_to_inchi("CccOc"))
    print(api_key)
