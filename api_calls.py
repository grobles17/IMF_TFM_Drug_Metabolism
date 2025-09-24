from chemspipy import ChemSpider
from pandas import DataFrame
from typing import Any

from rdkit import Chem
from typing import Any

cs = ChemSpider("UYH8H2lBS03Ow4vwVA1uxCw4MBHXhZk2xURiZmQa") #ChemSpider API

#list of tuples (DBoriginal, DB ID, CYPs, origname, name, InChI, SMILES)

def get_SMILES_chemspipy(missing_structures: DataFrame)-> list[tuple[Any, str, Any, Any, Any, None, Any]] | None:
    n = 0
    missing_smiles: list[tuple[Any, Any, Any, Any, Any, None, Any]] = []
    for i, drug in missing_structures.iterrows():
        results = cs.search(drug["Name"])
        for result in results:
            n += 1
            compound = cs.get_compound_info(result.record_id)
            missing_smiles.append((drug["DrugBank ID"], 
                                   f"DBX0{i}{n}", drug["CYPs"], 
                                   drug["Name"], compound["commonName"], 
                                   None, compound["smiles"]))
    return missing_smiles

def smiles_to_inchi(smile: str) -> Any:
    mol = Chem.MolFromSmiles(smile)
    smiles = Chem.MolToInchi(mol)
    return smiles

if __name__ == "__main__":
    pass
    