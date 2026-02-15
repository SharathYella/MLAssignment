
# app.py
# Minimal-deps Streamlit app for ML Assignment-2.
# Resilient: lazy-import scikit-learn and xgboost; if missing, show actionable UI message instead of crashing.

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------
# Page settings
# ------------------------
st.set_page_config(page_title="ML Assignment 2 — Breast Cancer", layout="wide")
st.title("Machine Learning Assignment‑2 — Classification on Breast Cancer Dataset")
st.caption("Minimal dependencies. If required libraries are missing, you’ll see a friendly instruction to fix the environment.")

# ------------------------
# Lazy import helpers
# ------------------------
def _import_sklearn():
    try:
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
            accuracy_score, precision_score, recall_score, f1_score,
            matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
        )
        return {
            "load_breast_cancer": load_breast_cancer,
            "train_test_split": train_test_split,
            "StandardScaler": StandardScaler,
            "Pipeline": Pipeline,
            "LogisticRegression": LogisticRegression,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "KNeighborsClassifier": KNeighborsClassifier,
            "GaussianNB": GaussianNB,
            "RandomForestClassifier": RandomForestClassifier,
            "metrics": {
                "accuracy_score": accuracy_score,
                "precision_score": precision_score,
                "recall_score": recall_score,
                "f1_score": f1_score,
                "matthews_corrcoef": matthews_corrcoef,
                "roc_auc_score": roc_auc_score,
                "confusion_matrix": confusion_matrix,
                "classification_report": classification_report,
            }
        }
    except ModuleNotFoundError:
        st.error(
            "❗ **scikit‑learn is not installed in this environment.**\n\n"
            "Please ensure a `requirements.txt` file (in the same folder as `app.py`) contains:\n"
            "```\nstreamlit\nscikit-learn\npandas\nnumpy\nxgboost\n```\n"
            "Then in Streamlit Cloud: **Manage app → Reboot**, and **Settings → Advanced → Clear all caches**.\n"
            "If your app is inside a subfolder, set the **Main file path** to that folder and place `requirements.txt` there as well."
        )
        st.stop()

def _import_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ModuleNotFoundError:
        # Not fatal; we’ll hide XGBoost if the lib is missing
        return None

SK = _import_sklearn()
XGBClassifier = _import_xgb()

# ------------------------
# Constants / paths
# ------------------------
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree.pkl",
    "kNN": MODEL_DIR / "knn.pkl",
    "Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
    "Random Forest": MODEL_DIR / "random_forest.pkl",
}
if XGBClassifier is not None:
    MODEL_FILES["XGBoost"] = MODEL_DIR / "xgboost.pkl"

# ------------------------
# Dataset
# ------------------------
@st.cache_resource(show_spinner=False)
def load_dataset():
    ds = SK["load_breast_cncer"
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, list(ds.feature_names), list(ds.target_names)

X, y, feature_names, target_names = load_dataset()
train_test_split = SK["train_test_split"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------
# Model builders
# ------------------------
def build_model(name: str):
    Pipeline = SK["Pipeline"]
    StandardScaler = SK["StandardScaler"]
    LogisticRegression = SK["LogisticRegression"]
    DecisionTreeClassifier = SK["DecisionTreeClassifier"]
    KNeighborsClassifier = SK["KNeighborsClassifier"]
    GaussianNB = SK["GaussianNB"]
    RandomForestClassifier = SK["RandomForestClassifier"]
