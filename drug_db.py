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

### Get SMILES using chemspipy (Already cached - Uncomment if you want to recall API) ###

# missing_SMILES: list[CompoundRecord] | None = get_SMILES_chemspipy(missing_structures)

# new_smiles_df: DataFrame = pd.DataFrame([
#     {
#         "DrugBank ID": t.drugbank_id,      
#         "CYPs": t.cyps,             
#         "Name": t.common_name,
#         "InChI": np.nan, 
#         "SMILES": t.smiles      
#     }
#     for t in missing_SMILES])

# ### Get InChI from SMILES (cached) 
# new_smiles_df["InChI"] = new_smiles_df["SMILES"].apply(smiles_to_inchi)
# ### Cache the new data to avoid recalling API
# new_smiles_df.to_csv("smiles_data_cache.csv", index=False)

new_smiles_df = pd.read_csv("smiles_data_cache.csv")
new_smiles_df["CYPs"] = new_smiles_df["CYPs"].apply(eval)

df_DrugBank = pd.concat([df_cyps_smd, new_smiles_df], ignore_index=True)
df_DrugBank_clean = df_DrugBank.dropna(subset=["SMILES"])

### InChI duplicates: keep the first occurrence; others are flagged for removal
duplicate_inchi = df_DrugBank_clean[df_DrugBank_clean.duplicated(subset=["InChI"], keep="first")]
### Gather DrugBank IDs of duplicates
ids_to_drop = duplicate_inchi["DrugBank ID"].tolist() 
### Remove duplicates from the main DataFrame
df_DrugBank_clean = df_DrugBank_clean[~df_DrugBank_clean["DrugBank ID"].isin(ids_to_drop)]

###CYP counter###
from collections import Counter

cyp_counter = Counter(cyp for cyps in df_DrugBank_clean["CYPs"] for cyp in cyps)

#Eliminate low frequency CYPs
THRESHOLD = 10
uncommon_cyps = [k for k, v in cyp_counter.items() if v < THRESHOLD]
df_DrugBank_curated = df_DrugBank_clean.copy()
df_DrugBank_curated["CYPs"] = df_DrugBank_clean["CYPs"].apply(
    lambda cyps: [cyp for cyp in cyps if cyp not in uncommon_cyps])

df_DrugBank_curated.to_csv("DrugBank_curated_df.csv", index=False) 

if __name__=="__main__":  
    ##### 
    print(df_DrugBank_clean["CYPs"].describe())
    df_cyp_counter = pd.DataFrame([(cyp, count) for cyp, count in cyp_counter.items()], 
                              columns= ["CYP", "DrugCount"])
    print(df_cyp_counter["DrugCount"].describe())
    print(df_cyp_counter.sort_values(by="DrugCount", ascending=False))