"""
Streamlit app for Assignment 2: Breast Cancer Wisconsin (Diagnostic)
classification model comparison.

Features:
  - Upload a test CSV (must match the schema of test_data.csv)
  - Select which trained model to evaluate
  - View evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  - View confusion matrix and classification report
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model", "saved")
TARGET_COL = "diagnosis"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in MODEL_FILES.items()
    }
    return scaler, feature_names, models


def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")
    st.title("Breast Cancer Wisconsin (Diagnostic) — Classifier Comparison")
    st.caption(
        "ML Assignment 2 — Logistic Regression, Decision Tree, kNN, "
        "Naive Bayes, and Random Forest compared on the same dataset."
    )

    scaler, feature_names, models = load_artifacts()

    st.sidebar.header("1. Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV with the same columns as test_data.csv "
        f"({len(feature_names)} features + '{TARGET_COL}' column)",
        type=["csv"],
    )

    st.sidebar.header("2. Select Model")
    model_name = st.sidebar.selectbox("Choose a classification model", list(models.keys()))

    if uploaded_file is None:
        st.info("Upload a test CSV from the sidebar to see model results. "
                 "You can use the provided test_data.csv.")
        return

    df = pd.read_csv(uploaded_file)

    missing_cols = set(feature_names + [TARGET_COL]) - set(df.columns)
    if missing_cols:
        st.error(f"Uploaded file is missing required columns: {sorted(missing_cols)}")
        return

    X = df[feature_names]
    y_true = df[TARGET_COL]
    X_scaled = scaler.transform(X)

    model = models[model_name]
    y_pred = model.predict(X_scaled)
    y_score = (
        model.predict_proba(X_scaled)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_scaled)
    )

    st.subheader(f"Results: {model_name}")

    metrics = compute_metrics(y_true, y_pred, y_score)
    cols = st.columns(len(metrics))
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, f"{value:.4f}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign (0)", "Malignant (1)"],
            yticklabels=["Benign (0)", "Malignant (1)"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col_right:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_true, y_pred, target_names=["Benign (0)", "Malignant (1)"],
            output_dict=True,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4))

    with st.expander("Compare all 5 models on this uploaded data"):
        rows = []
        for name, m in models.items():
            m_pred = m.predict(X_scaled)
            m_score = (
                m.predict_proba(X_scaled)[:, 1]
                if hasattr(m, "predict_proba")
                else m.decision_function(X_scaled)
            )
            rows.append({"Model": name, **compute_metrics(y_true, m_pred, m_score)})
        st.dataframe(pd.DataFrame(rows).round(4).set_index("Model"))

    with st.expander("View uploaded data"):
        st.dataframe(df)


if __name__ == "__main__":
    main()
