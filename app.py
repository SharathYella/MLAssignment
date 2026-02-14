!pip install seaborn

# app.py
# Streamlit UI that loads pre-trained .pkl models from ./model and evaluates on a holdout split.
# Also supports CSV upload for test-only predictions.

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Assignment-2 | Breast Cancer (Diagnostic)", layout="wide")

MODEL_DIR = Path("model")
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree":        MODEL_DIR / "decision_tree.pkl",
    "kNN":                  MODEL_DIR / "knn.pkl",
    "Naive Bayes":          MODEL_DIR / "naive_bayes.pkl",
    "Random Forest":        MODEL_DIR / "random_forest.pkl",
    "XGBoost":              MODEL_DIR / "xgboost.pkl",
}

# ---------- Utilities ----------

@st.cache_resource(show_spinner=False)
def load_dataset():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    feature_names = list(ds.feature_names)
    target_names = list(ds.target_names)  # ['malignant', 'benign'] (0, 1)
    return X, y, feature_names, target_names

@st.cache_resource(show_spinner=False)
def load_models(model_files: dict):
    loaded = {}
    for name, path in model_files.items():
        if not path.exists():
            # Give a helpful error for missing artifacts
            raise FileNotFoundError(
                f"Missing model file: {path}. "
                f"Run 'python generate_models.py' first to create .pkl files."
            )
        loaded[name] = joblib.load(path)
    return loaded

def compute_metrics(y_true, y_pred, y_proba_or_score=None):
    m = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "F1":        f1_score(y_true, y_pred),
        "MCC":       matthews_corrcoef(y_true, y_pred),
        "AUC":       np.nan,
    }
    if y_proba_or_score is not None:
        try:
            if hasattr(y_proba_or_score, "shape") and getattr(y_proba_or_score, "ndim", 1) == 2:
                m["AUC"] = roc_auc_score(y_true, y_proba_or_score[:, 1])
            else:
                m["AUC"] = roc_auc_score(y_true, y_proba_or_score)
        except Exception:
            m["AUC"] = np.nan
    return m

def predict_proba_if_supported(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None

# ---------- App ----------

st.title("Machine Learning Assignment–2: Classification Models on Breast Cancer (Diagnostic)")
st.caption("Loads pre-trained .pkl models. Includes metrics, confusion matrix, classification report, and CSV upload.")

X, y, feature_names, target_names = load_dataset()

with st.expander("ℹ️ Dataset details", expanded=False):
    st.write(
        f"**Instances**: {X.shape[0]}  |  "
        f"**Features**: {X.shape[1]}  |  "
        f"**Target classes**: {target_names}"
    )
    st.write("Binary classification dataset with 30 numeric features extracted from digitized images of a breast mass.")

# Train/test split for evaluation (on-the-fly)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Load all models from ./model
models = load_models(MODEL_FILES)

# Sidebar controls
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()), index=0)
show_class_report = st.sidebar.checkbox("Show classification report", value=True)
show_confusion    = st.sidebar.checkbox("Show confusion matrix", value=True)

# Evaluate selected model
model = models[model_name]
with st.spinner(f"Evaluating {model_name} ..."):
    y_pred  = model.predict(X_test)
    y_proba = predict_proba_if_supported(model, X_test)

metrics = compute_metrics(y_test, y_pred, y_proba)
metrics_df = (
    pd.DataFrame([metrics]).T.reset_index().rename(columns={"index": "Metric", 0: "Value"})
)
metrics_df["Value"] = metrics_df["Value"].apply(
    lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else v
)

st.subheader("Evaluation metrics (Test split)")
st.dataframe(metrics_df, width="stretch")

# Confusion matrix / report
if show_confusion:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    st.pyplot(fig)

if show_class_report:
    report_txt = classification_report(y_test, y_pred, target_names=target_names)
    st.text("Classification Report")
    st.code(report_txt, language="text")

# ---------- Upload test CSV ----------
st.subheader("Upload a test CSV (optional)")
st.caption(
    "Upload **only test data** (Free Streamlit tier is limited). "
    "CSV must contain the **30 feature columns** with the same names/order as the sklearn dataset."
)

# Provide a downloadable sample CSV
if st.button("Generate & download a sample_test.csv"):
    sample = X_test.iloc[:10].copy()
    csv_bytes = sample.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download sample_test.csv",
        data=csv_bytes,
        file_name="sample_test.csv",
        mime="text/csv",
    )

uploaded = st.file_uploader(
    "Choose CSV with the same 30 columns as the dataset", type=["csv"]
)
if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded)
        missing = [c for c in feature_names if c not in test_df.columns]
        extra   = [c for c in test_df.columns if c not in feature_names]

        if missing:
            st.error(f"Missing expected columns: {missing[:5]}{' ...' if len(missing)>5 else ''}")
        else:
            if extra:
                st.warning(f"Extra columns ignored: {extra[:5]}{' ...' if len(extra)>5 else ''}")
                test_df = test_df[feature_names]
            st.success("Columns validated. Running predictions…")

            preds = model.predict(test_df)
            probs = predict_proba_if_supported(model, test_df)

            out = pd.DataFrame({"prediction": preds.astype(int)})
            if probs is not None:
                if hasattr(probs, "shape") and getattr(probs, "ndim", 1) == 2 and probs.shape[1] > 1:
                    out["prob_benign(1)"] = probs[:, 1]
                else:
                    out["score"] = probs

            st.dataframe(out.head(20), width="stretch")
            st.download_button(
                "Download predictions",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Could not read CSV: {e}")

st.caption("Run this on **BITS Virtual Lab** and take a single screenshot showing metrics + confusion matrix for submission.")

