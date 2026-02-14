
# app.py
# Streamlit UI that loads/saves pre-trained .pkl models using Python's pickle.
# No joblib/matplotlib/seaborn dependency. If .pkl files are missing, trains and saves them automatically.

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import pickle

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

# Try to import XGBoost; app will still run if it's not available
XGB_AVAILABLE = True
try:
    from xgboost import XGBClassifier
except Exception:
    XGB_AVAILABLE = False

st.set_page_config(page_title="ML Assignment-2 | Breast Cancer (Diagnostic)", layout="wide")

# ----------------------- Constants -----------------------
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Logistic Regression":          MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree":                MODEL_DIR / "decision_tree.pkl",
    "kNN":                          MODEL_DIR / "knn.pkl",
    "Naive Bayes":                  MODEL_DIR / "naive_bayes.pkl",
    "Random Forest (Ensemble)":     MODEL_DIR / "random_forest.pkl",
    "XGBoost (Ensemble)":           MODEL_DIR / "xgboost.pkl",  # skipped if XGB not available
}

# ----------------------- Utilities -----------------------
@st.cache_resource(show_spinner=False)
def load_dataset():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    feature_names = list(ds.feature_names)
    target_names = list(ds.target_names)  # ['malignant', 'benign'] -> 0, 1
    return X, y, feature_names, target_names

def build_model_registry():
    """Return fresh, untrained estimators keyed by name."""
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
            # Scaling not strictly required; kept to mirror earlier metrics
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", GaussianNB())
        ]),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=42
        ),
    }
    if XGB_AVAILABLE:
        registry["XGBoost (Ensemble)"] = XGBClassifier(
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
    return registry

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

def pickle_load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def pickle_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_or_train_models(X_train, y_train):
    """
    Try to load each .pkl model; if missing, train and save it.
    Returns dict: name -> fitted estimator.
    """
    models = {}
    fresh_registry = build_model_registry()
    trained_any = False

    for name, default_path inThanks for the screenshot, Sharath. The new error:
