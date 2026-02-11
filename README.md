# MLAssignment
# ML Assignment 2 — Classification on Breast Cancer Wisconsin (Diagnostic)

**Student:** Yella Sharath  
**Task:** Implement 6 classification models, evaluate with specified metrics, and deploy a Streamlit app.

## 1) Problem Statement
Build, evaluate, and deploy multiple classification models for the **Breast Cancer Wisconsin (Diagnostic)** dataset to predict whether a tumor is **malignant (0)** or **benign (1)**.

## 2) Dataset Description
- Source: `sklearn.datasets.load_breast_cancer()`  
- Instances: **569**  
- Features: **30** (continuous numeric)  
- Target: Binary — `malignant (0)`, `benign (1)`  
- Split: 80/20 stratified train/test with `random_state=42`.

> Note: In the deployed app, the dataset loads from scikit-learn and models are trained on the fly for transparent, reproducible metrics.

## 3) Models Used
1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

## 4) Evaluation Metrics
For each model: **Accuracy, AUC, Precision, Recall, F1, MCC** computed on the test split.

### 4.1 Model Comparison Table (Test Set)

| ML Model Name              | Accuracy |   AUC  | Precision | Recall |   F1  |  MCC  |
|---------------------------:|:--------:|:------:|:---------:|:------:|:-----:|:-----:|
| Logistic Regression        | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861| 0.9623|
| Decision Tree              | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286| 0.8174|
| kNN                        | 0.9737   | 0.9884 | 0.9600    | 1.0000 | 0.9796| 0.9442|
| Naive Bayes                | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444| 0.8492|
| Random Forest (Ensemble)   | 0.9474   | 0.9937 | 0.9583    | 0.9583 | 0.9583| 0.8869|
| XGBoost (Ensemble)         | 0.9561   | 0.9950 | 0.9467    | 0.9861 | 0.9660| 0.9058|

*(Your run may differ slightly based on environment/library versions, but should be close.)*

### 4.2 Observations (Insights)
- **Logistic Regression**: Excellent linear separability on this dataset; highest **AUC (≈ 0.995)** and near-perfect precision/recall balance.  
- **Decision Tree**: Underperforms due to high variance; lacks the regularization/averaging of ensemble methods.  
- **kNN**: Very high **recall (≈ 1.0)** indicating very few false negatives; precision slightly lower due to neighborhood voting.  
- **Naive Bayes**: Competitive baseline with fast training; Gaussian assumption limits peak performance compared to LR/ensembles.  
- **Random Forest**: Strong performance; bagging reduces variance vs. single tree; near‑SOTA AUC.  
- **XGBoost**: Among top performers with excellent AUC and recall; boosting corrects difficult cases that LR/DT may miss.

## 5) How to Run Locally
```bash
# 1) Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Pre-train and save models to /model
python train_and_save.py

# 4) Launch the app
streamlit run app.py
``
