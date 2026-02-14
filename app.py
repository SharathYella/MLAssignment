
# app.py
# Streamlit UI that loads pre-trained .pkl models from ./model and evaluates on a holdout split.
# If .pkl artifacts are missing, it trains the models on the fly and saves them.

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

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
from xgboost import XGBClassifier

st.set_page_config(page_title="ML Assignment-2 | Breast Cancer (Diagnostic)", layout="wide")

# ----------------------- Constants -----------------------
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree":        MODEL_DIR / "decision_tree.pkl",
    "kNN":                  MODEL_DIR / "knn.pkl",
    "Naive Bayes":          MODEL_DIR / "naive_bayes.pkl",
    "Random Forest (Ensemble)": MODEL_DIR / "random_forest.pkl",
    "XGBoost (Ensemble)":       MODEL_DIR / "xgboost.pkl",
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
    return {
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
        ),
    }

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

def load_or_train_models(X_train, y_train):
    """
    Try to load each .pkl model; if missing, train and save it.
    Returns dict: name -> fitted estimator.
    """
    models = {}
    fresh_registry = build_model_registry()
    trained_any = False

    for name, path in MODEL_FILES.items():
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception:
                # If loading fails, retrain
                mdl = fresh_registry[name]
                mdl.fit(X_train, y_train)
                models[name] = mdl
                try:
                    joblib.dump(mdl, path)
                except Exception:
                    pass
                trained_any = True
        else:
            # Train and save
            mdl = fresh_registry[name]
            mdl.fit(X_train, y_train)
            models[name] = mdl
            try:
                joblib.dump(mdl, path)
            except Exception:
                pass
            trained_any = True

    if trained_any:
        st.info("Some model files were missing. Trained the models now and saved them to `model/*.pkl`.")
    return models

# ----------------------- App UI -----------------------
st.title("Machine Learning Assignment–2: Classification Models on Breast Cancer (Diagnostic)")
st.caption("Six models + metrics + confusion matrix + classification report + CSV upload. "
           "If `.pkl` files are missing, the app trains and saves them automatically.")

X, y, feature_names, target_names = load_dataset()

with st.expander("ℹ️ Dataset details", expanded=False):
    st.write(
        f"**Instances**: {X.shape[0]}  |  "
        f"**Features**: {X.shape[1]}  |  "
        f"**Target classes**: {target_names}"
    )
    st.write("Binary classification dataset with 30 numeric features extracted from digitized images of a breast mass.")

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Load or train models
with st.spinner("Loading models..."):
    models = load_or_train_models(X_train, y_train)

# Sidebar
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

# Confusion matrix without seaborn
if show_confusion:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    # Annotate cells
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center", color="black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(target_names); ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

if show_class_report:
    report_txt = classification_report(y_test, y_pred, target_names=target_names)
    st.text("Classification Report")
    st.code(report_txt, language="text")

# ----------------------- Upload test CSV -----------------------
st.subheader("Upload a test CSV (optional)")
st.caption(
    "Upload **only test data**. CSV must contain the **30 feature columns** "
    "with the same names/order as the sklearn dataset."
)

if st.button("Generate & download a sample_test.csv"):
    sample = X_test.iloc[:10].copy()
    st.download_button(
        label="Download sample_test.csv",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="sample_test.csv",
        mime="text/csv",
    )

uploaded = st.file_uploader("Choose CSV with the same 30 columns as the dataset", type=["csv"])
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

st.caption("If you are running this on BITS Virtual Lab, take one screenshot showing the metrics and confusion matrix for submission.")
