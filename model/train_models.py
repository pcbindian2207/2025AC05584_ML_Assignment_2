"""
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset and evaluates each on the same held-out test split:

  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbor Classifier
  4. Gaussian Naive Bayes Classifier
  5. Random Forest Classifier (Ensemble)

For each model, computes: Accuracy, AUC, Precision, Recall, F1, MCC.
Saves each fitted model (+ the shared StandardScaler) to model/saved/ as
.pkl files, and writes model/metrics_summary.csv with the comparison table.
"""

import os
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)
import joblib

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")

TARGET = "diagnosis"


def load_splits():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(ROOT, "test_data.csv"))
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    return X_train, y_train, X_test, y_test


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
    }


def evaluate(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_score = model.decision_function(X_test_scaled)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    os.makedirs(SAVED_DIR, exist_ok=True)

    X_train, y_train, X_test, y_test = load_splits()

    # Scale features once, shared across all models (kNN and Logistic Regression
    # are scale-sensitive; tree-based models are unaffected by scaling).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(SAVED_DIR, "scaler.pkl"))
    joblib.dump(list(X_train.columns), os.path.join(SAVED_DIR, "feature_names.pkl"))

    models = build_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        metrics = evaluate(model, X_test_scaled, y_test)
        results[name] = metrics

        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        joblib.dump(model, os.path.join(SAVED_DIR, filename))

        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    results_df = pd.DataFrame(results).T
    results_df.index.name = "ML Model Name"
    results_df = results_df.round(4)

    summary_path = os.path.join(os.path.dirname(__file__), "metrics_summary.csv")
    results_df.to_csv(summary_path)
    print(f"\nSaved comparison table: {summary_path}")

    with open(os.path.join(os.path.dirname(__file__), "metrics_summary.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
