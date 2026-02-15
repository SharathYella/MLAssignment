
# app.py
# Streamlit ML Assignment‑2 


from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)

# =========================
# Page Configuration
# =========================
st.set_page_config(page_title="ML Assignment 2 - Breast Cancer", layout="wide")
st.markdown(
    """
    <style>
    /* Subtle polish for tables and headers */
    .metric-table td, .metric-table th {font-size: 0.95rem;}
    .small-caption {color:#6c757d; font-size:0.9rem;}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Machine Learning Assignment‑ Multiple Classification Model")
st.subheader("Breast Cancer Classification Dashboard")

# =========================
# Data Loading
# =========================
@st.cache_resource(show_spinner=False)
def load_data():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    target_names = list(ds.target_names)
    feature_names = list(ds.feature_names)
    return X, y, target_names, feature_names

X, y, TARGET_NAMES, FEATURE_NAMES = load_data()

# One-time split for evaluation (consistent across models)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# Model Artifacts
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

def available_models():
    return [name for name, fn in MODEL_MAPPING.items() if (MODEL_DIR / fn).exists()]

@st.cache_resource(show_spinner=False)
def load_model(name: str):
    path = MODEL_DIR / MODEL_MAPPING[name]
    mdl = joblib.load(path)
    return mdl

# =========================
# Utilities
# =========================
def get_scores(model, X_):
    """
    Return (score_vector, score_type)
    score_type: 'proba' (probability for class 1), 'decision' (raw score), or None
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X_)[:, 1], "proba"
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X_), "decision"
        except Exception:
            pass
    return None, None

def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC Score": roc_auc_score(y_true, y_score) if y_score is not None else np.nan,
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Matthews Correlation Coefficient (MCC Score)": matthews_corrcoef(y_true, y_pred),
    }

def metrics_to_df(mdict):
    df = pd.DataFrame(list(mdict.items()), columns=["Metric", "Value"])
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}" if not np.isnan(v) else "0.0000"
        return v
    df["Value"] = df["Value"].apply(fmt)
    return df

def plot_confusion_matrix(cm: np.ndarray, labels_actual, labels_pred, normalize=False):
    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=150)
    show = cm.copy().astype(float)
    title = "Confusion Matrix"
    if normalize:
        # Normalize by row (true class)
        with np.errstate(all="ignore"):
            row_sums = show.sum(axis=1, keepdims=True)
            show = np.divide(show, row_sums, out=np.zeros_like(show), where=row_sums!=0)
        title = "Confusion Matrix (Normalized by Actual)"

    im = ax.imshow(show, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1], labels=[f"Pred {l}" for l in labels_pred])
    ax.set_yticks([0, 1], labels=[f"Actual {l}" for l in labels_actual])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    vmax = show.max() if show.size else 1
    for i in range(show.shape[0]):
        for j in range(show.shape[1]):
            val = show[i, j]
            display = f"{val:.2f}" if normalize else f"{int(cm[i,j])}"
            color = "white" if val > 0.6 * vmax else "black"
            ax.text(j, i, display, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=270, labelpad=10)
    plt.tight_layout()
    return fig

def plot_roc_pr_curves(y_true, y_score):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), dpi=150)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {roc_auc:.4f}")
    axes[0].plot([0,1], [0,1], color="gray", lw=1, linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    axes[1].plot(recall, precision, color="#d62728", lw=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall Curve")

    plt.tight_layout()
    return fig

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("Model Settings")
present = available_models()
if not present:
    st.error("No model artifacts found under `./model`. Please add `.pkl` files first.")
    st.stop()

model_choice = st.sidebar.selectbox("Select ML Model", present, index=0)
model = load_model(model_choice)

# Advanced controls
normalize_cm = st.sidebar.checkbox("Normalize confusion matrix", value=False)
show_curves = st.sidebar.checkbox("Show ROC & PR curves (if available)", value=True)

# Threshold control (if probabilities or decision scores available)
y_score_raw, score_type = get_scores(model, X_test)
custom_threshold = None
if score_type == "proba":
    custom_threshold = st.sidebar.slider("Decision threshold (for probability)", 0.0, 1.0, 0.50, 0.01)
elif score_type == "decision":
    st.sidebar.caption("Using decision function; threshold fixed at 0.")
else:
    st.sidebar.caption("This model doesn't expose probabilities/scores.")

# =========================
# Dataset Info (collapsible)
# =========================
with st.expander("ℹ️ Dataset details", expanded=False):
    st.write(
        f"- **Instances:** {X.shape[0]}  \n"
        f"- **Features:** {X.shape[1]}  \n"
        f"- **Target classes:** {TARGET_NAMES}  \n"
        f"- **Train/Test split:** 80/20 (stratified, random_state=42)"
    )

# =========================
# EVALUATION — Metrics first (as requested)
# =========================
# Apply threshold if applicable
if score_type == "proba" and custom_threshold is not None:
    y_pred_eval = (y_score_raw >= custom_threshold).astype(int)
    y_score_for_auc = y_score_raw
elif score_type == "decision":
    y_pred_eval = (y_score_raw >= 0.0).astype(int)
    y_score_for_auc = y_score_raw
else:
    y_pred_eval = model.predict(X_test)
    y_score_for_auc = y_score_raw  # may be None

metrics_dict = compute_metrics(y_test, y_pred_eval, y_score_for_auc)
metrics_df = metrics_to_df(metrics_dict)

st.write(f"### Performance: {model_choice}")
st.caption(
    "Evaluation on the 20% hold‑out test split. "
    + (f"Decision threshold = **{custom_threshold:.2f}**" if score_type == "proba" and custom_threshold is not None else "")
)
st.dataframe(metrics_df, use_container_width=True, column_config={"Metric": {"width": 360}})

# =========================
# CONFUSION MATRIX — below the metrics table
# =========================
st.write("#### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred_eval)
row_labels = [TARGET_NAMES[0], TARGET_NAMES[1]]
col_labels = [TARGET_NAMES[0], TARGET_NAMES[1]]
fig_cm = plot_confusion_matrix(cm, row_labels, col_labels, normalize=normalize_cm)
st.pyplot(fig_cm, clear_figure=True)

# =========================
# CLASSIFICATION REPORT
# =========================
st.write("#### Classification Report")
st.code(classification_report(y_test, y_pred_eval, target_names=TARGET_NAMES), language="text")

# =========================
# OPTIONAL: Curves
# =========================
if show_curves and y_score_for_auc is not None:
    st.write("#### ROC & Precision–Recall Curves")
    fig_curves = plot_roc_pr_curves(y_test, y_score_for_auc)
    st.pyplot(fig_curves, clear_figure=True)

# =========================
# LEADERBOARD — Compare all available models in ./model
# =========================
with st.expander("🏆 Compare all models (Leaderboard)", expanded=False):
    def evaluate_model_quick(name):
        mdl = load_model(name)
        preds = mdl.predict(X_test)
        scr, _stype = get_scores(mdl, X_test)
        auc_val = roc_auc_score(y_test, scr) if scr is not None else np.nan
        return {
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "AUC": auc_val,
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "MCC": matthews_corrcoef(y_test, preds),
        }

    if st.button("Compute Leaderboard"):
        rows = [evaluate_model_quick(nm) for nm in present]
        lb = pd.DataFrame(rows)
        for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
            lb[col] = lb[col].apply(lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else "0.0000")
        lb = lb.sort_values(by="F1", ascending=False, ignore_index=True)
        st.dataframe(lb, use_container_width=True)

# =========================
# CSV PREDICTIONS
# =========================
st.divider()
st.write("### Predict on New Data (CSV)")
st.caption(
    "Upload **test‑only** CSV with the **same 30 feature columns** as the sklearn dataset.  \n"
    "Extra columns will be ignored; missing columns are filled with 0."
)

col_dl, _ = st.columns(2)
with col_dl:
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
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        extra = [c for c in df.columns if c not in FEATURE_NAMES]
        if missing:
            st.warning(f"Missing expected columns filled with 0: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if extra:
            st.info(f"Ignoring extra columns: {extra[:10]}{' ...' if len(extra) > 10 else ''}")

        df_aligned = df.reindex(columns=FEATURE_NAMES, fill_value=0)
        preds = model.predict(df_aligned)

        out = pd.DataFrame({"prediction": preds.astype(int)})

        # Probabilities & custom threshold labels (if available)
        scrs, stype = get_scores(model, df_aligned)
        if scrs is not None:
            if stype == "proba":
                out["prob_class_1"] = scrs
                if custom_threshold is not None:
                    out["pred_at_thr"] = (scrs >= custom_threshold).astype(int)
            else:
                out["score"] = scrs
                out["pred_at_thr"] = (scrs >= 0.0).astype(int)

        st.write("#### Prediction Results (first 20 rows)")
        st.dataframe(out.head(20), use_container_width=True)
        st.download_button(
            "Download predictions.csv",
            out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Error processing CSV: {e}")

# =========================
# Friendly footer
# =========================
st.caption(
    "Tip: Use the sidebar to toggle normalization and curves, and to adjust the decision threshold "
    "when the selected model exposes probabilities."
)

