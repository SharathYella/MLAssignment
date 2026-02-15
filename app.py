
# app.py
# Streamlit app for ML Assignment‑2 — minimal dependencies (streamlit, scikit-learn, pandas, numpy, xgboost)
# Robust: works even if PKL files were created with joblib elsewhere (we retrain & save with pickle when needed).

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
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

# xgboost (optional guard)
try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="ML Assignment 2 — Breast Cancer", layout="wide")
st.title("Machine Learning Assignment‑2 — Classification on Breast Cancer Dataset")
st.caption("Fast, lightweight app with minimal dependencies. If a model file is missing or incompatible, it will be retrained on the fly and saved as a pickle.")

# ------------------------
# Constants
# ------------------------
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree.pkl",
    "kNN": MODEL_DIR / "knn.pkl",
    "Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
    "Random Forest": MODEL_DIR / "random_forest.pkl",
    "XGBoost": MODEL_DIR / "xgboost.pkl",
}
if not _XGB_AVAILABLE:
    # If xgboost can't import on this machine, hide the option even if a file exists.
    MODEL_FILES.pop("XGBoost", None)

# ------------------------
# Dataset
# ------------------------
@st.cache_resource(show_spinner=False)
def load_dataset():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, list(ds.feature_names), list(ds.target_names)

X, y, feature_names, target_names = load_dataset()

# One split used across the session (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------
# Model builders
# ------------------------
def build_model(name: str):
    if name == "Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))
        ])
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    if name == "kNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ])
    if name == "Naive Bayes":
        # GaussianNB does not need scaling; keep it simple for speed.
        return GaussianNB()
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=42)
    if name == "XGBoost" and _XGB_AVAILABLE:
        return XGBClassifier(
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
    raise ValueError(f"Unknown or unavailable model: {name}")

# ------------------------
# Robust artifact loader (pickle only)
# If loading fails (e.g., file created with joblib), we will retrain & overwrite as pickle.
# ------------------------
def load_model_or_retrain(name: str, path: Path):
    # Try to load with pickle
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            # file exists but incompatible → retrain & overwrite below
            pass

    # Train on FULL dataset to create a final artifact (fast — dataset is small)
    model = build_model(name)
    model.fit(X, y)
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        st.warning(f"Could not save model artifact for '{name}': {e}")
    return model

# ------------------------
# Helpers
# ------------------------
def get_scores(model, X_):
    """Return probability/score for class 1 if available, else None."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X_)[:, 1]
        except Exception:
            return None
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X_)
        except Exception:
            return None
    return None

def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC Score": (roc_auc_score(y_true, y_score) if y_score is not None else np.nan),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC Score": matthews_corrcoef(y_true, y_pred),
    }

# ------------------------
# Sidebar
# ------------------------
st.sidebar.header("Controls")

available_names = list(MODEL_FILES.keys())
if not available_names:
    st.error("No models available. Install xgboost or remove it from the list, then rerun.")
    st.stop()

selected_name = st.sidebar.selectbox("Select model", available_names, index=0)

# Load or retrain the chosen model (and persist as pickle)
model_path = MODEL_FILES[selected_name]
model = load_model_or_retrain(selected_name, model_path)

# ------------------------
# Evaluate on held-out test split
# ------------------------
with st.spinner(f"Evaluating {selected_name} ..."):
    y_pred = model.predict(X_test)
    y_score = get_scores(model, X_test)

metrics = compute_metrics(y_test, y_pred, y_score)

st.subheader(f"Performance Metrics — {selected_name}")
cols = st.columns(6)
for i, (k, v) in enumerate(metrics.items()):
    if isinstance(v, float):
        cols[i].metric(k, f"{v:.4f}")
    else:
        cols[i].metric(k, str(v))

st.divider()
c1, c2 = st.columns(2)

with c1:
    st.write("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {target_names[0]}", f"Actual {target_names[1]}"],
        columns=[f"Pred {target_names[0]}", f"Pred {target_names[1]}"],
    )
    st.dataframe(cm_df, use_container_width=True)

with c2:
    st.write("#### Classification Report")
    st.code(classification_report(y_test, y_pred, target_names=target_names), language="text")

# ------------------------
# Upload CSV for predictions
# ------------------------
st.divider()
st.subheader("Predict on New Data (CSV)")
st.caption(
    "Upload **test-only** data. CSV must contain the **same 30 feature columns** as the sklearn dataset. "
    "Column names will be aligned automatically (missing columns filled with 0)."
)

left, _ = st.columns(2)
with left:
    if st.button("Download sample_test.csv (first 10 rows from X_test)"):
        sample = X_test.iloc[:10].copy()
        st.download_button(
            "Download sample_test.csv",
            sample.to_csv(index=False).encode("utf-8"),
            file_name="sample_test.csv",
            mime="text/csv",
        )

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Align columns (order + fill missing with 0); ignore extras
    missing = [c for c in feature_names if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_names]
    if missing:
        st.warning(f"Missing expected columns filled with 0: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if extra:
        st.info(f"Ignoring extra columns: {extra[:10]}{' ...' if len(extra) > 10 else ''}")

    df_aligned = df.reindex(columns=feature_names, fill_value=0)

    preds = model.predict(df_aligned)
    scores = get_scores(model, df_aligned)

    out = pd.DataFrame({"prediction": preds.astype(int)})
    if scores is not None:
        out["prob_class_1"] = scores

    st.write("#### Prediction Results (first 20 rows)")
    st.dataframe(out.head(20), use_container_width=True)
    st.download_button(
        "Download predictions.csv",
        out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

# ------------------------
# Notes for deployment
# ------------------------
st.caption(
    "Tip: If you previously saved artifacts with joblib and don't have the `joblib` package installed, "
    "this app will retrain the selected model and overwrite the artifact as a standard pickle for portability."
)
