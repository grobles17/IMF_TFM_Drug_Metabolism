import pandas as pd
from chemspipy import ChemSpider

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
print(missing_structures)

cs = ChemSpider("LtPG7M2FaS9fbwJuCpUCD84WI1QDJ09fuBbr0CN9") #ChemSpider API

print(df_cyps_smd.columns)
missing_smiles: list[tuple[str]] = []
#list of tuples (DBoriginal, DB ID, CYPs, origname, name, InChI, SMILES)
n = 0
for i, drug in missing_structures.iterrows():
    for result in cs.search(drug["Name"]):
        n += 1
        compound = cs.get_compound_info(result.record_id)
        missing_smiles.append((drug["DrugBank ID"], f"DBX0{i}{n}", drug["CYPs"], drug["Name"], compound["commonName"], None, compound["smiles"]))
        print(f"For {drug}, result: {cs.get_compound_info(result.record_id)["commonName"]}\nwith SMILES {cs.get_compound_info(result.record_id)["smiles"]}")
print(missing_smiles)

###CYP counter###
cyp_counter: dict[str, int] = {}
for cyps in drug_CYPs.values():
    for cyp in cyps:
        # setdefault initializes the count to 0 if the key doesn't exist
        cyp_counter.setdefault(cyp, 0)
        cyp_counter[cyp] += 1

max_cyp = max(cyp_counter, key=cyp_counter.get)
#print(f"{max_cyp} = {cyp_counter[max_cyp]}")
