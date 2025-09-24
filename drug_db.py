import pandas as pd
import numpy as np
from pandas import DataFrame
from api_calls import *

#drugs with enzyme data
csvfile_enzyme = pd.read_csv("./DataBase/drug_enzyme_db_all.csv", delimiter= ",", header= 0, encoding="utf-8")
#drugs with smiles/Inchi data
csvfile_smiles = pd.read_csv("./DataBase/structure_smiles_links_all.csv", delimiter= ",", header= 0, encoding="utf-8")
#all drugs with external references and drug type
csvfile_drugs = pd.read_csv("./DataBase/drug_links_all.csv", delimiter= ",", header=0, encoding="utf-8")

#Get all IDs of SmallMolecules to ignore proteins and peptides
small_molecule_ids = csvfile_drugs.loc[
    csvfile_drugs["Drug Type"] == "SmallMoleculeDrug", "DrugBank ID"]

def name_to_cypcode(CypName: str) -> str | None:
    """
    Convert a cytochrome P450 enzyme name into its standard CYP code.

    Example:
        "Cytochrome P450 1A2" -> "CYP1A2"

    Args:
        CypName (str): Full enzyme name in plain text.

    Returns:
        str | None: Standardized CYP code if input contains "Cytochrome P450",
        otherwise None.
    """
    if "Cytochrome P450" in CypName:
        code: str = "CYP" + CypName.split()[2].strip(",")
        return code
    return

### Create dfs ###
drug_CYPs: dict[str, list[str]] = {}
for _, row in csvfile_enzyme.iterrows():
    drug_id = row["DrugBank ID"]
    cyp_code = name_to_cypcode(row["UniProt Name"])
    
    if cyp_code:  # only add valid CYPs
        drug_CYPs.setdefault(drug_id, []).append(cyp_code)

df_cyps = pd.DataFrame(
    [(drug, cyps) for drug, cyps in drug_CYPs.items()],
    columns=["DrugBank ID", "CYPs"])

df_cyps = pd.merge(
    df_cyps,
    csvfile_smiles[["DrugBank ID", "Name", "InChI", "SMILES"]],
    on="DrugBank ID",
    how="left"   # keep all drugs from df_cyps, fill missing InChI/SMILES with NaN
)

df_cyps_smd = df_cyps[df_cyps["DrugBank ID"].isin(small_molecule_ids)]
missing_structures = df_cyps_smd.loc[df_cyps_smd["InChI"].isnull()]

### Get SMILES using chemspipy (Cached)
# missing_SMILES: list[tuple[str | None]] | None = get_SMILES_chemspipy(missing_structures)

# new_smiles_df: DataFrame = pd.DataFrame([
#     {
#         "DrugBank ID": t[1],      
#         "CYPs": t[2],             
#         "Name": t[4],             
#         "InChI": np.nan if t[5] is None else t[5], 
#         "SMILES": t[6]      
#     }
#     for t in missing_SMILES])

# new_smiles_df.to_csv("smiles_data_cache.csv", index=False)

df = pd.read_csv("smiles_data_cache.csv")

### Get InChI from SMILES
    
df["InChI"] = df["SMILES"].apply(smiles_to_inchi)

df_DrugBank = pd.concat([df_cyps_smd, df], ignore_index=True)
df_DrugBank_clean = df_DrugBank.dropna(subset=["SMILES"])

# duplicate_names = df_DrugBank_clean[df_DrugBank_clean.duplicated(subset=["Name"], keep=False)]
# duplicate_inchi = df_DrugBank_clean[df_DrugBank_clean.duplicated(subset=["InChI"], keep=False)]

ids_of_duplicates = ["DBX0106610", "DBX0107111", "DBX0126213"]
df_DrugBank_clean = df_DrugBank_clean[~df_DrugBank_clean["DrugBank ID"].isin(ids_of_duplicates)]

df_DrugBank_clean.to_csv("DrugBank_curated_df.csv", index=False)

###CYP counter###
cyp_counter: dict[str, int] = {}
for cyps in drug_CYPs.values():
    for cyp in cyps:
        # setdefault initializes the count to 0 if the key doesn't exist
        cyp_counter.setdefault(cyp, 0)
        cyp_counter[cyp] += 1


if __name__=="__main__":    
    max_cyp = max(cyp_counter, key=cyp_counter.get)
    print(f"{max_cyp} = {cyp_counter[max_cyp]}")
    print(df_DrugBank_clean.describe())