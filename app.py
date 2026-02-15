# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, roc_auc_score, 
    confusion_matrix, classification_report
)

# --- Configuration ---
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Machine Learning Assignment-2")
st.write("Implementation of 6 classification models on the Breast Cancer dataset.")

# --- Step 1: Dataset [cite: 27-30] ---
@st.cache_resource
def get_data():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, list(ds.target_names)

X, y, targets = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- Step 6b: Model Selection [cite: 34-39, 92] ---
MODEL_DIR = Path("model")
MODEL_MAP = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

st.sidebar.header("Model Settings")
choice = st.sidebar.selectbox("Select ML Model", list(MODEL_MAP.keys()))

@st.cache_resource
def load_model(name):
    path = MODEL_DIR / MODEL_MAP[name]
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model(choice)

if model is None:
    st.error(f"Missing model file: {MODEL_MAP[choice]} in the 'model/' folder.")
    st.stop()

# --- Step 6c: Display Metrics [cite: 40-46, 93] ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0,
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC Score": matthews_corrcoef(y_test, y_pred)
}

st.subheader(f"Evaluation Metrics: {choice}")
cols = st.columns(6)
for i, (m_name, m_val) in enumerate(metrics.items()):
    cols[i].metric(m_name, f"{m_val:.4f}")

# --- Step 6d: Confusion Matrix / Report [cite: 94] ---
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.write("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=targets, columns=targets))
with col2:
    st.write("#### Classification Report")
    st.code(classification_report(y_test, y_pred, target_names=targets))

# --- Step 6a: Test Data Upload [cite: 91] ---
st.divider()
st.subheader("Predict on New Data")
st.info("Upload a CSV with exactly 30 features (matching Breast Cancer dataset).")
uploaded = st.file_uploader("Upload test data (CSV)", type="csv")

if uploaded:
    test_df = pd.read_csv(uploaded)
    # Ensure columns match training features
    test_df = test_df.reindex(columns=X.columns, fill_value=0)
    
    preds = model.predict(test_df)
    
    # FIXED: Corrected prediction DataFrame format to avoid SyntaxError
    out = pd.DataFrame({"prediction": preds.astype(int)})
    
    st.write("#### Results")
    st.dataframe(out)
    st.download_button("Download Predictions", out.to_csv(index=False), "predictions.csv")
