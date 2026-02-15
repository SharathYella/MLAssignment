# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Use joblib to match train.py
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, roc_auc_score, 
    confusion_matrix, classification_report
)

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(page_title="ML Assignment 2 - Breast Cancer", layout="wide")
st.title("Machine Learning Assignment-2")
st.subheader("Breast Cancer Classification Dashboard")

# ------------------------
# Dataset Loader
# ------------------------
@st.cache_resource
def load_data():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, list(ds.target_names)

X, y, target_names = load_data()
# Using same split as evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ------------------------
# Model Selection & Loading
# ------------------------
MODEL_DIR = Path("model")
MODEL_MAPPING = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Select ML Model", list(MODEL_MAPPING.keys()))

@st.cache_resource
def load_model_artifact(name):
    path = MODEL_DIR / MODEL_MAPPING[name]
    if path.exists():
        # Using joblib.load to match joblib.dump in train.py
        return joblib.load(path)
    return None

model = load_model_artifact(model_choice)

if model is None:
    st.error(f"Error: {MODEL_MAPPING[model_choice]} not found. Please ensure your models are in the 'model/' folder.")
    st.stop()

# ------------------------
# Display Evaluation Metrics
# ------------------------
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred),
    "AUC Score": roc_auc_score(y_test, y_score) if y_score is not None else 0.0
}

st.write(f"### Performance: {model_choice}")
cols = st.columns(6)
for i, (name, val) in enumerate(metrics.items()):
    cols[i].metric(name, round(val, 4))

# ------------------------
# Confusion Matrix & Report
# ------------------------
col1, col2 = st.columns(2)
with col1:
    st.write("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=target_names, columns=target_names))

with col2:
    st.write("#### Classification Report")
    st.code(classification_report(y_test, y_pred, target_names=target_names))

# ------------------------
# Test Data Prediction
# ------------------------
st.divider()
st.write("### Custom Prediction")
uploaded_file = st.file_uploader("Upload test CSV data (30 features)", type="csv")

if uploaded_file:
    try:
        test_df = pd.read_csv(uploaded_file)
        # Match columns to training data
        test_df = test_df.reindex(columns=X.columns, fill_value=0)
        
        preds = model.predict(test_df)
        
        # Corrected format: generated as requested
        out = pd.DataFrame({"prediction": preds.astype(int)})
        
        st.write("#### Prediction Results")
        st.dataframe(out)
        st.download_button("Download Predictions", out.to_csv(index=False), "predictions.csv")
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
