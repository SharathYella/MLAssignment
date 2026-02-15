
# app.py
# ML Assignment‑2

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
)

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment‑2 — Fast Streamlit App")
st.caption("Optimized for fast load & low memory.")

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
# LIGHTWEIGHT DATA LOAD
# ------------------------
@st.cache_resource
def load_dataset():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, list(ds.feature_names), list(ds.target_names)

# ------------------------
# FAST MODEL LOADING
# ------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

X, y, feature_names, target_names = load_dataset()

# Train/test split only once
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------
# Sidebar
# ------------------------
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))

# Load only the selected model (saves loading time)
model = load_model(MODEL_FILES[model_name])

# ------------------------
# Prediction for evaluation
# ------------------------
y_pred = model.predict(X_test)

# Predict proba if available
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)[:, 1]
else:
    y_proba = None

# ------------------------
# Compute metrics
# ------------------------
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
}

st.subheader("📊 Evaluation Metrics")
dfm = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
dfm["Value"] = dfm["Value"].apply(lambda x: round(x, 4) if isinstance(x, float) else x)
st.dataframe(dfm, use_container_width=True)

# ------------------------
# Confusion Matrix (simple, no seaborn)
# ------------------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(pd.DataFrame(cm,
                      index=["Actual 0", "Actual 1"],
                      columns=["Pred 0", "Pred 1"]))

# ------------------------
# Classification report
# ------------------------
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, target_names=target_names)
st.code(report)

# ------------------------
# CSV upload (minimal)
# ------------------------
st.subheader("Upload test CSV (optional)")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    test_df = pd.read_csv(uploaded)

    # Validate columns
    missing = [c for c in feature_names if c not in test_df.columns]
    if missing:
        st.error(f"Missing columns: {missing[:5]}")
    else:
        test_df = test_df[feature_names]
        preds = model.predict(test_df)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(test_df)[:, 1]
            output = pd.DataFrame({"Prediction": preds, "Probability(class=1)": prob})
        else:
            output = pd.DataFrame({"Prediction": preds})

        st.dataframe(output.head(20))

        st.download_button(
            "Download predictions",
            output.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            mime="text/csv"
        )
``
