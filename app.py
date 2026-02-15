
# app.py 
# ML Assignment‑2

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="ML Assignment 2 — Breast Cancer", layout="wide")
st.title("Machine Learning Assignment‑2 — Classification on Breast Cancer Dataset")
st.caption("Lightweight app optimized for Streamlit Community Cloud.")

# ------------------------
# Constants
# ------------------------
MODEL_DIR = Path("model")
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree.pkl",
    "kNN": MODEL_DIR / "knn.pkl",
    "Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
    "Random Forest": MODEL_DIR / "random_forest.pkl",
    "XGBoost": MODEL_DIR / "xgboost.pkl",
}

# ------------------------
# Cached loaders
# ------------------------
@st.cache_resource(show_spinner=False)
def load_dataset():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, list(ds.feature_names), list(ds.target_names)

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    return joblib.load(path)

# ------------------------
# Utilities
# ------------------------
def get_proba_or_score(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            return None
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            return None
    return None

# ------------------------
# Data preparation (single split)
# ------------------------
X, y, feature_names, target_names = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------
# Sidebar controls
# ------------------------
st.sidebar.header("Controls")
# Keep only those models whose files actually exist, to avoid deployment errors
available_models = {name: p for name, p in MODEL_FILES.items() if p.exists()}
if not available_models:
    st.error(
        "No model artifacts found under ./model. Please run `python generate_models.py` "
        "to create the .pkl files before deploying."
    )
    st.stop()

model_name = st.sidebar.selectbox("Select model", list(available_models.keys()), index=0)

# ------------------------
# Load selected model (fast)
# ------------------------
model_path = available_models[model_name]
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {model_path}\nError: {e}")
    st.stop()

# ------------------------
# Evaluate on test split
# ------------------------
with st.spinner(f"Evaluating {model_name} ..."):
    y_pred = model.predict(X_test)
    y_score = get_proba_or_score(model, X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred),
    "AUC": (roc_auc_score(y_test, y_score) if y_score is not None else np.nan),
}

st.subheader("Evaluation Metrics (Test Split)")
metrics_df = (
    pd.DataFrame([metrics]).T.reset_index().rename(columns={"index": "Metric", 0: "Value"})
)
metrics_df["Value"] = metrics_df["Value"].apply(
    lambda v: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
)
st.dataframe(metrics_df, use_container_width=True)

# Confusion matrix table (no heavy plotting)
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
st.dataframe(cm_df, use_container_width=True)

# Classification report
st.subheader("Classification Report")
report_txt = classification_report(y_test, y_pred, target_names=target_names)
st.code(report_txt, language="text")

# ------------------------
# CSV upload for test data
# ------------------------
st.subheader("Upload a test CSV (optional)")
st.caption(
    "Upload **only test data**. The CSV must have the same 30 feature columns "
    "as the sklearn dataset (names must match)."
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Generate sample_test.csv (first 10 rows of X_test)"):
        sample = X_test.iloc[:10].copy()
        st.download_button(
            label="Download sample_test.csv",
            data=sample.to_csv(index=False).encode("utf-8"),
            file_name="sample_test.csv",
            mime="text/csv",
        )

uploaded = st.file_uploader("Choose CSV", type=["csv"])
if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Validate columns
    missing = [c for c in feature_names if c not in test_df.columns]
    extra = [c for c in test_df.columns if c not in feature_names]

    if missing:
        st.error(f"Missing expected columns: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    else:
        if extra:
            st.warning(f"Extra columns will be ignored: {extra[:10]}{' ...' if len(extra) > 10 else ''}")
        test_df = test_df[feature_names]
        preds = model.predict(test_df)
        scores = get_proba_or_score(model, test_df)

        out = pd.DataFrame({"prediction": preds.astype(int)})
        if scores is not None:
            # If scores are probabilities for class 1, keep as is; else just name 'score'
            if hasattr(model, "predict_proba"):
                out["prob_class_1"] = scores
            else:
                out["score"] = scores

        st.dataframe(out.head(20), use_container_width=True)
        st.download_button(
            "Download predictions",
            out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
