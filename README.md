# Predicting Drug Metabolism with Machine Learning
### A Reproducible Benchmark for CYP450 Metabolism Prediction

> **Master's Thesis – MSc in Big Data**
> **Author:** Gonzalo Robles Criado
> **Institution:** IMF Business School
> **Grade:** Highest Distinction

---

## Project Overview

Drug metabolism is one of the major causes of failure during pharmaceutical development. Determining **which Cytochrome P450 (CYP450) enzymes metabolize a candidate drug** is essential for assessing toxicity, drug-drug interactions and pharmacokinetics, but experimental characterization is expensive and usually performed late in development.

This project investigates whether **Machine Learning** can accurately predict CYP450 metabolism directly from molecular structure.

Rather than proposing a single model, this repository provides a **fully reproducible benchmark** comparing multiple molecular representations and machine learning algorithms under an identical evaluation framework.

The project was developed as the final Master's Thesis for the MSc in Big Data.

---

# Objectives

The benchmark evaluates two fundamental questions:

### 1️⃣ Which molecular representation works best?

The following representations are compared:

- Morgan Fingerprints (ECFP6)
- MolE embeddings
- ChemBERTa embeddings
- Author-trained InChI Transformer embeddings *(exploratory)*

---

### 2️⃣ Which machine learning algorithm performs best?

Each representation is evaluated using:

- Logistic Regression
- Random Forest
- XGBoost

alongside a Majority Baseline predictor.

All experiments use exactly the same:

- train/test split
- cross-validation folds
- hyperparameter optimisation
- threshold optimisation
- evaluation metrics

making every comparison fair and reproducible.

---

# Dataset

The benchmark is based on a curated DrugBank dataset containing

- **1,368 drug-like molecules**
- **13 CYP450 isoenzymes**
- Multi-label annotations
- Canonical molecular structures (SMILES/InChI)

---

# Benchmark Workflow

```text
DrugBank
    │
    ▼
Data Curation
    │
    ▼
Molecular Representation
    ├── Morgan Fingerprints
    ├── MolE
    ├── ChemBERTa
    └── InChI Transformer
            │
            ▼
Machine Learning
    ├── Logistic Regression
    ├── Random Forest
    ├── XGBoost
    └── Majority Baseline
            │
            ▼
Cross Validation
            │
            ▼
Threshold Optimisation
            │
            ▼
Independent Test Evaluation
            │
            ▼
Benchmark Comparison
```

---

# 📂 Repository Structure

```text
IMF_TFM_Drug_Metabolism
│
├── 📄 Memorias/
│      Final thesis (PDF)
│
├── 📁 DataBases/
│      Original and curated datasets
│
├── 📁 ETL/
│      Data acquisition
│      Cleaning
│      Database curation
│      DrugBank processing
│
├── 📁 Representations/
│      Morgan fingerprint generation
│      ChemBERTa embeddings
│      MolE embeddings
│      InChI embedding extraction
│
├── 📁 InChI_Transformer/
│      Tokenizer training
│      Transformer pretraining
│      Embedding extraction
│
├── 📁 Main_Pipeline/
│      Complete benchmarking framework
│
│      ├── models/
│      ├── results/
│      ├── splits/
│      └── evaluation/
│
├── 📁 Figures/
│      Figures used in the thesis
│
├── 📁 Utils/
│      Helper scripts
│
└── README.md
```

---

# Where to Start

If you're new to the repository, this is the recommended order:

```text
README
   │
   ▼
📄 Thesis (Memorias/)
   │
   ▼
ETL
   │
   ▼
Representations
   │
   ▼
Main Pipeline
   │
   ▼
Results
```

---

# Main Components

## 📁 ETL

Contains every script required to reproduce the dataset.

Includes:

- DrugBank parsing
- identifier recovery
- duplicate removal
- missing value handling
- final curated dataset generation

---

## 📁 Representations

Transforms molecular structures into numerical features.

Implemented representations:

- Morgan Fingerprints
- MolE
- ChemBERTa
- InChI embeddings

Each representation is generated independently so they can be benchmarked under identical conditions.

---

## 📁 InChI Transformer

Contains the exploratory experiment presented in the thesis.

Pipeline:

```text
1. Build tokenizer
        │
        ▼
2. Pretrain Transformer
        │
        ▼
3. Generate molecular embeddings
        │
        ▼
4. Benchmark against SMILES-based models
```

---

## 📁 Main Pipeline

The core of the project.

Responsible for

- loading representations
- loading labels
- training models
- hyperparameter search
- threshold optimisation
- independent evaluation
- exporting results

Every experiment follows the exact same protocol.

---

# Evaluation

Models are evaluated using

- Matthews Correlation Coefficient (Macro MCC)
- Micro F1
- Hamming Loss

Macro MCC is used as the primary metric because it is considerably more robust to the strong class imbalance present in CYP450 prediction.

---

# Reproducibility

This repository was designed to be fully reproducible.

Every experiment uses

- identical train/test partitions
- fixed random seeds
- identical CV folds
- identical evaluation metrics
- identical optimisation protocol

This allows representation and model comparisons without introducing methodological bias.

---

# Thesis

The complete methodology, benchmark and discussion can be found in

📄 **/Memorias/Predicting Novel Drug Metabolism with Machine Learning - Gonzalo Robles**

---

# Technologies

- Python
- RDKit
- Scikit-learn
- XGBoost
- HuggingFace Transformers
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

# 📜 Citation

If this repository contributes to your work, please consider citing the accompanying Master's Thesis.

---

# 🤝 Acknowledgements

This work was developed as part of the **Master's in Big Data** at **IMF Business School**.
