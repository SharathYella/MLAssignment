# app.py
# Streamlit app for Breast Cancer Wisconsin (Diagnostic) — 6 ML models + metrics + CM + upload
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# XGBoost
from xgboost import XGBClassifier

st.set_page_config(page_title="ML Assignment-2 | Breast Cancer (Diagnostic)", layout="wide")

# ---------- Utility ----------

@st.cache_resource(show_spinner=False)
def load_dataset():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    feature_names = list(ds.feature_names)
    target_names = list(ds.target_names)  # ['malignant', 'benign'] (0=malignant, 1=benign)
    return X, y, feature_names, target_names

def build_model_registry():
    # Keep choices consistent with evaluation I computed for the README
    registry = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),  # included for parity with computed metrics
            ("clf", GaussianNB())
        ]),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=42
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=4
        )
    }
    return registry

def compute_metrics(y_true, y_pred, y_proba_or_score=None):
    metrics = {}
    metrics["Accuracy"]  = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred)
    metrics["Recall"]    = recall_score(y_true, y_pred)
    metrics["F1"]        = f1_score(y_true, y_pred)
    metrics["MCC"]       = matthews_corrcoef(y_true, y_pred)
    if y_proba_or_score is not None:
        try:
            # if 2D array, use [:,1]
            if hasattr(y_proba_or_score, "shape") and len(y_proba_or_score.shape) == 2:
                metrics["AUC"] = roc_auc_score(y_true, y_proba_or_score[:, 1])
            else:
                metrics["AUC"] = roc_auc_score(y_true, y_proba_or_score)
        except Exception:
            metrics["AUC"] = np.nan
    else:
        metrics["AUC"] = np.nan
    return metrics

def predict_proba_if_supported(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None

# ---------- App UI ----------

st.title("Machine Learning Assignment–2: Classification Models on Breast Cancer (Diagnostic)")
st.caption("Includes: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost — metrics, confusion matrix, and CSV upload.")

X, y, feature_names, target_names = load_dataset()

with st.expander("ℹ️ Dataset details", expanded=False):
    st.write(f"**Instances**: {X.shape[0]}  |  **Features**: {X.shape[1]}  |  **Target classes**: {target_names}")
    st.write("This is a binary classification dataset with 30 numeric features extracted from digitized images of a breast mass.")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Sidebar controls
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox(
    "Choose a model",
    list(build_model_registry().keys()),
    index=0
)
show_class_report = st.sidebar.checkbox("Show classification report", value=True)
show_confusion    = st.sidebar.checkbox("Show confusion matrix", value=True)

# Train/evaluate selected model
models = build_model_registry()
model = models[model_name]

with st.spinner(f"Training {model_name} ..."):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = predict_proba_if_supported(model, X_test)

metrics = compute_metrics(y_test, y_pred, y_proba)
metrics_df = pd.DataFrame([metrics]).T.reset_index()
metrics_df.columns = ["Metric", "Value"]
metrics_df["Value"] = metrics_df["Value"].apply(lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else v)

st.subheader("Evaluation metrics (Test split)")
st.dataframe(metrics_df, use_container_width=True)

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
st.caption("Free Streamlit tier has limited resources, so upload **only test data**. The CSV must contain the **30 feature columns** of the dataset in the same order.")

# Provide a downloadable sample CSV
if st.button("Generate & download a sample_test.csv"):
    # use some rows from X_test
    sample = X_test.iloc[:10].copy()
    csv_bytes = sample.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download sample_test.csv",
        data=csv_bytes,
        file_name="sample_test.csv",
        mime="text/csv"
    )

uploaded = st.file_uploader("Choose CSV with the same 30 columns as the dataset", type=["csv"])
if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded)
        missing = [c for c in feature_names if c not in test_df.columns]
        extra   = [c for c in test_df.columns if c not in feature_names]
        if missing:
            st.error(f"Missing expected columns: {missing[:5]}{' ...' if len(missing)>5 else ''}")
        elif extra:
            st.warning(f"Extra columns in file will be ignored: {extra[:5]}{' ...' if len(extra)>5 else ''}")
            test_df = test_df[feature_names]
        else:
            st.success("Columns validated. Running predictions…")
            preds = model.predict(test_df)
            probs = predict_proba_if_supported(model, test_df)

            out = pd.DataFrame({
                "prediction": preds.astype(int)
            })
            if probs is not None:
                if hasattr(probs, "shape") and len(probs.shape) == 2 and probs.shape[1] > 1:
                    out["prob_benign(1)"] = probs[:, 1]
                else:
                    out["score"] = probs

            st.dataframe(out.head(20), use_container_width=True)
            csv_buf = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions", csv_buf, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Could not read CSV: {e}")

st.caption("Tip: For the assignment screenshot, run this app on **BITS Virtual Lab**, then take a single screenshot showing the metrics and confusion matrix.")
