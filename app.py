
# app.py
# Streamlit app for ML Assignment‑2 — minimal dependencies (streamlit, scikit-learn, pandas, numpy, xgboost)
# Robust behaviors:
#   • Lazy-import scikit-learn so the UI can render a helpful error if it's missing.
#   • Load model artifacts with pickle; if not present or incompatible, retrain & save fresh pickle.
#   • Skip XGBoost if the library isn't installed (even if the file exists).

from pathlib import Path
import pickle
import importlib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="ML Assignment 2 — Breast Cancer", layout="wide")
st.title("Machine Learning Assignment‑2 — Classification on Breast Cancer Dataset")
st.caption("Fast, minimal‑dependency app. If a model file is missing or incompatible, it will be re‑trained and saved as a pickle.")

# ------------------------
# Environment diagnostics (helps you see what's actually installed on Streamlit Cloud)
# ------------------------
def pkg_version(dist_name: str) -> str:
    try:
        try:
            # Python 3.8+
            import importlib.metadata as md
        except Exception:
            # Python <3.8 backport
            import importlib_metadata as md  # type: ignore
        return md.version(dist_name)
    except Exception:
        return "not installed"

with st.expander("ℹ️ Environment diagnostics", expanded=False):
    st.write(
        {
            "numpy": pkg_version("numpy"),
            "pandas": pkg_version("pandas"),
            "scikit-learn": pkg_version("scikit-learn"),
            "xgboost": pkg_version("xgboost"),
            "streamlit": pkg_version("streamlit"),
        }
    )
    st.caption(
        "If a package shows 'not installed', ensure your `requirements.txt` (same folder as `app.py`) "
        "lists it, then **Manage app → Reboot** and **Settings → Advanced → Clear all caches**."
    )

# ------------------------
# Lazy import utilities (avoid crashing at import time)
# ------------------------
def require(module: str):
    """Import a module by name or show a helpful Streamlit error and stop."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        st.error(
            f"Required library **{module}** is not installed.\n\n"
            "Make sure your `requirements.txt` (same folder as `app.py`) contains:\n"
            "```\nstreamlit\nscikit-learn\npandas\nnumpy\nxgboost\n```\n"
            "Then go to **Manage app → Reboot** and **Settings → Advanced → Clear all caches**."
        )
        st.stop()

# Optional XGBoost availability check
def xgb_available() -> bool:
    try:
        importlib.import_module("xgboost")
        return True
    except ModuleNotFoundError:
        return False

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
if not xgb_available():
    # Hide XGBoost if the library isn't installed, even if a file exists.
    MODEL_FILES.pop("XGBoost", None)

# ------------------------
# Dataset
# ------------------------
@st.cache_resource(show_spinner=False)
def load_dataset():
    datasets = require("sklearn.datasets")
    ds = datasets.load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, list(ds.feature_names), list(ds.target_names)

X, y, feature_names, target_names = load_dataset()

# Train/test split (once per session)
@st.cache_resource(show_spinner=False)
def get_split(X_df: pd.DataFrame, y_s: pd.Series):
    model_selection = require("sklearn.model_selection")
    return model_selection.train_test_split(
        X_df, y_s, test_size=0.2, stratify=y_s, random_state=42
    )

X_train, X_test, y_train, y_test = get_split(X, y)

# ------------------------
# Model builders
# ------------------------
def build_model(name: str):
    pipeline = require("sklearn.pipeline")
    preprocessing = require("sklearn.preprocessing")
    linear_model = require("sklearn.linear_model")
    tree = require("sklearn.tree")
    neighbors = require("sklearn.neighbors")
    naive_bayes = require("sklearn.naive_bayes")
    ensemble = require("sklearn.ensemble")

    if name == "Logistic Regression":
        return pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                ("clf", linear_model.LogisticRegression(max_iter=500, solver="lbfgs")),
            ]
        )
    if name == "Decision Tree":
        return tree.DecisionTreeClassifier(random_state=42)
    if name == "kNN":
        return pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                ("clf", neighbors.KNeighborsClassifier(n_neighbors=7)),
            ]
        )
    if name == "Naive Bayes":
        return naive_bayes.GaussianNB()
    if name == "Random Forest":
        return ensemble.RandomForestClassifier(n_estimators=300, random_state=42)
    if name == "XGBoost" and xgb_available():
        xgb = require("xgboost")
        return xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=4,
        )
    raise ValueError(f"Unknown or unavailable model: {name}")

# ------------------------
# Load artifact (pickle) or retrain if missing/incompatible
# ------------------------
def load_model_or_retrain(name: str, path: Path):
    # Try to load with pickle first
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            # Incompatible artifact → fall through to retrain
            pass

    # Retrain on FULL dataset to create a fresh portable artifact
    model = build_model(name)
    model.fit(X, y)
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        st.warning(f"Could not save model artifact for '{name}': {e}")
    return model

# ------------------------
# Helper: get score/probability
# ------------------------
def get_scores(model, X_):
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

# ------------------------
# Metrics
# ------------------------
def compute_metrics(y_true, y_pred, y_score):
    metrics = require("sklearn.metrics")
    return {
        "Accuracy": metrics.accuracy_score(y_true, y_pred),
        "AUC Score": (metrics.roc_auc_score(y_true, y_score) if y_score is not None else np.nan),
        "Precision": metrics.precision_score(y_true, y_pred),
        "Recall": metrics.recall_score(y_true, y_pred),
        "F1 Score": metrics.f1_score(y_true, y_pred),
        "MCC Score": metrics.matthews_corrcoef(y_true, y_pred),
    }

# ------------------------
# Sidebar
# ------------------------
st.sidebar.header("Controls")

available_names = list(MODEL_FILES.keys())
if not available_names:
    st.error(
        "No available models. Install `xgboost` or remove it from the list, then reboot the app."
    )
    st.stop()

selected_name = st.sidebar.selectbox("Select model", available_names, index=0)

# Load or retrain the selected model
model_path = MODEL_FILES[selected_name]
model = load_model_or_retrain(selected_name, model_path)

# ------------------------
# Evaluate on held‑out test split
# ------------------------
with st.spinner(f"Evaluating {selected_name} ..."):
    y_pred = model.predict(X_test)
    y_score = get_scores(model, X_test)

metrics = compute_metrics(y_test, y_pred, y_score)

st.subheader(f"Performance Metrics — {selected_name}")
cols = st.columns(6)
for i, (k, v) in enumerate(metrics.items()):
    cols[i].metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))

st.divider()
c1, c2 = st.columns(2)

with c1:
    st.write("#### Confusion Matrix")
    metrics_mod = require("sklearn.metrics")
    cm = metrics_mod.confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {target_names[0]}", f"Actual {target_names[1]}"],
        columns=[f"Pred {target_names[0]}", f"Pred {target_names[1]}"],
    )
    st.dataframe(cm_df, use_container_width=True)

with c2:
    st.write("#### Classification Report")
    metrics_mod = require("sklearn.metrics")
    st.code(metrics_mod.classification_report(y_test, y_pred, target_names=target_names), language="text")

# ------------------------
# Upload CSV for predictions
# ------------------------
st.divider()
st.subheader("Predict on New Data (CSV)")
st.caption(
    "Upload **test‑only** data. CSV must contain the **same 30 feature columns** as the sklearn dataset. "
    "Columns will be aligned automatically (missing columns filled with 0; extra columns ignored)."
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
# Footer note
# ------------------------
st.caption(
    "If the app shows a missing‑package error, confirm `requirements.txt` is next to `app.py`, "
    "then **Manage app → Reboot** and **Settings → Advanced → Clear all caches** to rebuild the environment."
)
