import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from embedder import smiles_to_morgan_fingerprint
from collections import Counter

#1. Load dataset
df = pd.read_csv("DrugBank_curated_df.csv")
# Parse CYPs column (stringified list → Python list)
df["CYPs"] = df["CYPs"].apply(eval)

# 2. Build label matrix
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["CYPs"]) #20 item binary list
cyp_labels = mlb.classes_

# 3. Build feature matrix
fps = [smiles_to_morgan_fingerprint(smi) for smi in df["SMILES"]]
X = np.vstack(fps)
print(Y)
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# 5. Baseline model (XGBoost one-vs-rest)
# Use "hist" for constructing the trees, with early stopping enabled.
clf = XGBClassifier(tree_method="hist", early_stopping_rounds=2, learning_rate = 0.1, n_estimators = 500)
# Fit the model, test sets are used for early stopping.
def fit_and_score(estimator, X_train, X_test, y_train, y_test):
    """Fit the estimator on the train set and score it on both sets"""
    estimator.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    train_score = estimator.score(X_train, y_train)
    test_score = estimator.score(X_test, y_test)

    return estimator, train_score, test_score

estimator, train_scores, test_scores = fit_and_score(clf, X_train, X_test, y_train, y_test)
print(f"train = {train_scores}, test = {test_scores}")

# 6. Evaluate
y_pred = clf.predict_proba(X_test)  # returns list of arrays, one per CYP
# for row in y_pred:
#     print(cyp_labels[list(row).index(max(row))]) #print most probable CYP
# roc_scores = {}
# pr_scores = {}
# for i, label in enumerate(cyp_labels):
#     y_true = y_test[:, i]
#     y_score = y_pred[i][:, 1] if isinstance(y_pred, list) else y_pred[:, i]
#     roc_scores[label] = roc_auc_score(y_true, y_score)
#     pr_scores[label] = average_precision_score(y_true, y_score)

# print("ROC-AUC per CYP:", roc_scores)
# print("PR-AUC per CYP:", pr_scores)
# print("Macro ROC-AUC:", np.mean(list(roc_scores.values())))
# print("Macro PR-AUC:", np.mean(list(pr_scores.values())))