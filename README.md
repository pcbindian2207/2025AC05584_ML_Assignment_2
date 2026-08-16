# ML Assignment 2 — Breast Cancer Classification (Model Comparison + Streamlit App)

## a. Problem Statement

Breast cancer diagnosis from digitized fine-needle aspirate (FNA) images of a
breast mass is a binary classification problem: given a set of measured
characteristics of the cell nuclei in the image, predict whether the mass is
**Malignant (1)** or **Benign (0)**.

This assignment implements and compares five classification models —
Logistic Regression, Decision Tree, k-Nearest Neighbor (kNN), Gaussian Naive
Bayes, and Random Forest (Ensemble) — on the same dataset, evaluates each
using a common set of metrics, and exposes the comparison through an
interactive Streamlit web application.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Data Set (WDBC)
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
**Instances:** 569
**Features:** 30 real-valued input features (+ 1 binary target column `diagnosis`)
**Class distribution:** 357 Benign (B), 212 Malignant (M) — binary classification

Features are computed from a digitized image of an FNA of a breast mass and
describe characteristics of the cell nuclei present in the image. Ten base
characteristics are measured per nucleus, and for each, the **mean**,
**standard error (se)**, and **worst** (mean of the three largest values) are
computed, giving 10 × 3 = 30 features:

`radius, texture, perimeter, area, smoothness, compactness, concavity,
concave_points, symmetry, fractal_dimension` (each suffixed `_mean`, `_se`,
`_worst`).

The target column `diagnosis` is encoded as `1 = Malignant`, `0 = Benign`.

The full dataset (`model/../data/breast_cancer_full.csv`) was split 80/20
(stratified on the target) into a training set (`data/train_data.csv`, 455
rows) used to fit all models, and a held-out test set (`test_data.csv`, 114
rows) used for evaluation and for the Streamlit app.

## c. GitHub Repository Link

`https://github.com/pcbindian2207/2025AC05584_ML_Assignment_2`

The repository contains:
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `README.md` — this file
- `test_data.csv` — held-out test data (used by the Streamlit app)
- `model/` — training script (`train_models.py` and an equivalent executed
  notebook `train_models.ipynb`), data preparation script
  (`prepare_data.py`), and saved model files (`model/saved/*.pkl`)

## d. Models Used

All five models were trained on the same 80% training split of the Breast
Cancer Wisconsin dataset and evaluated on the same 20% held-out test split
(`test_data.csv`). Features were standardized (zero mean, unit variance)
before fitting, since kNN (a distance-based, "lazy learning" method) and
Logistic Regression are scale-sensitive.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble) | 0.9649 | 0.9942 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

*(Malignant = positive class (1) for Precision/Recall/F1/AUC. Reproduce these
numbers by running `python model/prepare_data.py` then
`python model/train_models.py`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Fits a linear decision boundary in the standardized feature space (sigmoid of `w0 + wᵀx`, thresholded at 0.5). Highest AUC (0.9960) and joint-highest accuracy (0.9649) of all five models — the 30 nuclear features are close to linearly separable in log-odds space, which matches the original WDBC paper's finding that the classes are linearly separable. |
| Decision Tree | Weakest model on every metric (Accuracy 0.9298, AUC 0.9246, MCC 0.8492). A single unpruned tree (default Gini-index splits) overfits the 455-row training set and does not generalize as well as the other models on this test split — consistent with the bias-variance tradeoff where a fully grown tree has low bias but high variance. |
| kNN | Strong performance (Accuracy 0.9561, AUC 0.9823) with k = 5 on standardized features. As a lazy/instance-based learner, it relies entirely on local distances between the query point and training instances, and benefits from the good class separability of this dataset, though it trails Logistic Regression and Random Forest slightly on Recall. |
| Naive Bayes | Lowest Accuracy (0.9211) and Recall (0.8571) among all models, despite a high AUC (0.9891). The Gaussian Naive Bayes conditional-independence assumption does not hold well here, since many of the 30 features (e.g. `radius_mean`, `perimeter_mean`, `area_mean`) are highly correlated by construction, which hurts a model that assumes class-conditional feature independence. |
| Random Forest (Ensemble) | Joint-best Accuracy (0.9649) and best AUC among the non-Logistic-Regression models (0.9942), with perfect Precision (1.0000) — it never predicts a false Malignant on this test split. As a bagging ensemble of decision trees (bootstrap-sampled rows and randomly sampled features per split), it corrects the single Decision Tree's overfitting and reduces variance while keeping the tree's low bias. |
| **Overall Winner for your dataset?** | **Logistic Regression**, on the strength of the highest AUC (0.9960) and the best Recall (0.9286) among the top-tier models — Recall matters most in a cancer-diagnosis setting, since a false negative (missed malignant case) is far costlier than a false positive. Random Forest is a very close second (identical Accuracy, higher Precision, slightly lower Recall and AUC). |

## Streamlit App Features

- **Dataset upload (CSV):** upload a test CSV matching the schema of
  `test_data.csv` (30 feature columns + `diagnosis` label column).
- **Model selection dropdown:** choose any of the 5 trained models.
- **Evaluation metrics display:** Accuracy, AUC, Precision, Recall, F1, MCC
  shown for the selected model on the uploaded data.
- **Confusion matrix & classification report:** both displayed side by side,
  plus an expandable comparison of all 5 models on the same uploaded data.

## Live Links

- **GitHub Repository:** `https://github.com/pcbindian2207/2025AC05584_ML_Assignment_2`
- **Live Streamlit App:** `<TODO>`

## How to Run Locally

```bash
pip install -r requirements.txt
python model/prepare_data.py     # regenerates data/ and test_data.csv from data_raw/wdbc.data
python model/train_models.py     # trains all 5 models, saves to model/saved/, writes model/metrics_summary.csv
streamlit run app.py
```
