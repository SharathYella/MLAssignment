
# app.py
# Streamlit ML Assignment-2 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

# =========================
# Page Configuration
# =========================
st.set_page_config(page_title="ML Assignment 2 - Breast Cancer", layout="wide")
st.title("Machine Learning - Multiple Classification Models")
st.subheader("Breast Cancer Classification Dashboard")

# =========================
# Dataset Loader
# =========================
@st.cache_resource(show_spinner=False)
def load_data():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, list(ds.target_names), list(ds.feature_names)

X, y, target_names, feature_names = load_data()

# One-time split for evaluation view
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# Model Selection & Loading
# =========================
MODEL_DIR = Path("model")
MODEL_MAPPING = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Select ML Model", list(MODEL_MAPPING.keys()))

@st.cache_resource(show_spinner=False)
def load_model_artifact(name: str):
    path = MODEL_DIR / MODEL_MAPPING[name]
    if not path.exists():
        return None, path
    try:
        model = joblib.load(path)
        return model, path
    except Exception as e:
        st.error(f"Failed to load model at {path}.\nError: {e}")
        return None, path

model, artifact_path = load_model_artifact(model_choice)
if model is None:
    st.error(
        f"Missing model file: **{MODEL_MAPPING[model_choice]}** in the 'model/' folder.\n\n"
        f"Expected path: `{artifact_path}`"
    )
    st.stop()

# =========================
# Evaluation & Metrics
# =========================
y_pred = model.predict(X_test)
y_score = None
if hasattr(model, "predict_proba"):
    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_score = None
elif hasattr(model, "decision_function"):
    try:
        y_score = model.decision_function(X_test)
    except Exception:
        y_score = None

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred),
    "AUC Score": roc_auc_score(y_test, y_score) if y_score is not None else np.nan,
}

st.write(f"### Performance: {model_choice}")
cols = st.columns(6)
for i, (name, val) in enumerate(metrics.items()):
    # Guard against nan display for AUC if score is None
    display_val = 0.0 if (isinstance(val, float) and np.isnan(val)) else val
    cols[i].metric(name, f"{display_val:.4f}" if isinstance(display_val, float) else str(display_val))

# =========================
# Confusion Matrix (Matplotlib Heatmap)
# =========================
cm = confusion_matrix(y_test, y_pred)
# Label names (align to target_names from dataset)
row_labels = [f"{target_names[0]}", f"{target_names[1]}"]   # Actual
col_labels = [f"{target_names[0]}", f"{target_names[1]}"]   # Predicted

c1, c2 = st.columns(2)

with c1:
    st.write("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=150)
    im = ax.imshow(cm, cmap="Blues")

    # Ticks & labels
    ax.set_xticks([0, 1], labels=[f"Pred {lab}" for lab in col_labels])
    ax.set_yticks([0, 1], labels=[f"Actual {lab}" for lab in row_labels])

    # Cell text
    max_val = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > 0.6 * max_val else "black"
            ax.text(j, i, val, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    # Axes labels and colorbar
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=10)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

with c2:
    st.write("#### Classification Report")
    st.code(classification_report(y_test, y_pred, target_names=target_names), language="text")

# =========================
# Custom CSV Prediction
# =========================
st.divider()
st.write("### Custom Prediction")
st.caption(
    "Upload **test-only** CSV with the **same 30 feature columns** as the sklearn dataset. "
    "Extra columns are ignored; missing columns will be filled with 0."
)

uploaded_file = st.file_uploader("Upload test CSV data (30 features)", type=["csv"])
if uploaded_file:
    try:
        test_df = pd.read_csv(uploaded_file)

        # Align columns to training set feature order
        missing = [c for c in feature_names if c not in test_df.columns]
        extra = [c for c in test_df.columns if c not in feature_names]
        if missing:
            st.warning(
                f"Missing expected columns filled with 0: {missing[:10]}{' ...' if len(missing) > 10 else ''}"
            )
        if extra:
            st.info(
                f"Ignoring extra columns: {extra[:10]}{' ...' if len(extra) > 10 else ''}"
            )

        test_df = test_df.reindex(columns=feature_names, fill_value=0)

        preds = model.predict(test_df)
        out = pd.DataFrame({"prediction": preds.astype(int)})

        # Add probability, if available
        if hasattr(model, "predict_proba"):
            try:
                out["prob_class_1"] = model.predict_proba(test_df)[:, 1]
            except Exception:
                pass

        st.write("#### Prediction Results (first 20 rows)")
        st.dataframe(out.head(20), use_container_width=True)
        st.download_button(
            "Download Predictions",
            out.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
