import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, hamming_loss

from collections import Counter

import ast

#1. Load dataset
def load_representation(name):
    return pd.read_csv(f"{name}_output_representation.tsv", sep="\t")
def load_dataset(path):
    """Load the dataset from a CSV file and return a DataFrame."""
    df = pd.read_csv(path)
    return df
df = load_dataset("DrugBank_curated_df.csv")
# Parse CYPs column (stringified list → Python list)
df["CYPs"] = df["CYPs"].apply(ast.literal_eval)

# 2. Build label matrix
def build_labels(df):
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["CYPs"])
    return Y, mlb

# 3. Hyperparameters grid search space
param_grid = {
    "estimator__learning_rate": [0.05, 0.1, 0.3],
    "estimator__max_depth": [3, 6, 9],
    "estimator__n_estimators": [100, 200],
}
# 4. Train/test split
def split_data(X, Y):
    return train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42
    )

# 5. Baseline model (XGBoost one-vs-rest)
# Use "hist" for constructing the trees, with early stopping enabled.
def build_model(params=None):
    base = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        **(params or {})
    )
    
    return OneVsRestClassifier(base)
clf = build_model()
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
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    return {
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_micro": f1_score(y_test, y_pred, average="micro"),
        "hamming_loss": hamming_loss(y_test, y_pred)
    }

# 7. Save the model
import joblib
joblib.dump(model, "model_ecfp6.pkl")

def main():
    df = load_dataset(...)
    Y, mlb = build_labels(df)
    
    representations = ["morgan", "chemberta", "mole", "inchi"]
    
    results = []
    
    for rep in representations:
        X = load_representation(rep)
        
        X_train, X_test, y_train, y_test = split_data(X, Y)
        
        # Default model
        model_default = build_model()
        model_default.fit(X_train, y_train)
        metrics_default = evaluate(model_default, X_test, y_test)
        
        # Tuned model
        model = build_model()
        grid = GridSearchCV(model, param_grid, cv=3, scoring="f1_macro")
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        metrics_tuned = evaluate(best_model, X_test, y_test)
        
        results.append({
            "representation": rep,
            **metrics_default,
            **{f"tuned_{k}": v for k, v in metrics_tuned.items()}
        })
    
    pd.DataFrame(results).to_csv("results.csv", index=False)