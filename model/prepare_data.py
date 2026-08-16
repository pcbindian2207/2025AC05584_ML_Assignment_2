"""
Loads the raw UCI Wisconsin Diagnostic Breast Cancer (WDBC) data file,
attaches proper column names, splits into train/test, and writes:
  - data/breast_cancer_full.csv   (full labeled dataset)
  - data/train_data.csv           (80% - used to fit models)
  - test_data.csv                 (20% - held out; this is what the
                                    Streamlit app and submission use)

Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data_raw", "wdbc.data")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

# Official WDBC attribute names (10 base features x mean/se/worst = 30 features)
BASE_FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
COLUMNS = (
    ["id", "diagnosis"]
    + [f"{f}_mean" for f in BASE_FEATURES]
    + [f"{f}_se" for f in BASE_FEATURES]
    + [f"{f}_worst" for f in BASE_FEATURES]
)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH, header=None, names=COLUMNS)
    df = df.drop(columns=["id"])

    # diagnosis: M = malignant, B = benign -> binary target (1 = malignant, 0 = benign)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    full_path = os.path.join(DATA_DIR, "breast_cancer_full.csv")
    df.to_csv(full_path, index=False)
    print(f"Wrote full dataset: {full_path} ({df.shape[0]} rows, {df.shape[1]} cols)")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["diagnosis"]
    )

    train_path = os.path.join(DATA_DIR, "train_data.csv")
    train_df.to_csv(train_path, index=False)
    print(f"Wrote train dataset: {train_path} ({train_df.shape[0]} rows)")

    test_path = os.path.join(PROJECT_ROOT, "test_data.csv")
    test_df.to_csv(test_path, index=False)
    print(f"Wrote test dataset: {test_path} ({test_df.shape[0]} rows)")

    print("\nClass balance (full):")
    print(df["diagnosis"].value_counts())


if __name__ == "__main__":
    main()
