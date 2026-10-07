# Insurance Claim Fraud Detection — Ensemble Learning System

> **Final-Year Computer Science & Engineering Project (Revision 2)**  
> An interactive multi-page web application and machine learning engine for detecting vehicle insurance claim fraud using **Stacking Ensemble Learning** (Random Forest, XGBoost, Isolation Forest, and a Logistic Regression Meta-Learner).

---

## 📌 Project Overview

Vehicle insurance claim fraud presents a major challenge to the insurance industry. Traditional single-model solutions often struggle with class imbalance, complex feature interactions, or non-linear fraud patterns. 

This project implements a **Stacking Ensemble System (Stacked Generalization)** that combines three distinct base machine learning algorithms:
1. **Random Forest Classifier**: A bagging ensemble of 250 decision trees capturing non-linear interactions across policyholder attributes.
2. **XGBoost Classifier**: A sequential gradient boosted decision tree model using `scale_pos_weight=15.69` to achieve high fraud recall (89.73%).
3. **Isolation Forest**: An unsupervised anomaly detection model that isolates unusual claim profiles without relying on class labels.

The out-of-fold probability outputs from these three base models serve as inputs to a **Logistic Regression Meta-Learner**, which makes the final prediction and categorizes the claim into a 3-tier action recommendation (`High-confidence Fraud`, `High-confidence Legit`, or `Disagreement - Human Review`).

---

## 🧭 Multi-Tab Single-Page Application (SPA) Structure

The web application features a responsive, tabbed navigation bar allowing users to seamlessly transition between high-level prediction and deep-dive model analysis without page reloads:

| Navigation Tab | View ID | Description & Features |
| :--- | :--- | :--- |
| **Predict** | `#page-predict` | Main evaluation dashboard with 31-feature claim form, preset sample buttons, 3 base model summary cards, model confidence comparison bar chart, and headline Meta-Learner verdict. |
| **Random Forest** | `#page-rf` | Deep-dive analysis view: Stat cards (Accuracy 91.60%, ROC-AUC 0.8424), hyperparameters table (`n_estimators=250`, `min_samples_leaf=2`), `RandomizedSearchCV` tuning details, ROC curve, PR curve, and live prediction banner. |
| **XGBoost** | `#page-xgb` | Deep-dive analysis view: Stat cards (Recall 89.73%, ROC-AUC 0.8339), hyperparameters table (`n_estimators=150`, `scale_pos_weight=15.69`), class-weighting details, ROC curve, PR curve, and live prediction banner. |
| **Isolation Forest** | `#page-iso` | Deep-dive analysis view: Stat cards (Accuracy 86.93%, Contamination 5.98%), contamination anchor details, raw score thresholding, anomaly score distribution histogram, and live prediction banner. |

---

## 📂 Project Repository Inventory

| File / Folder Name | Type | Purpose & Role in System |
| :--- | :--- | :--- |
| `models/ensemble_rf_model.pkl` | Model Artifact | Trained Random Forest Classifier binary object used by app.py. |
| `models/ensemble_boosting_model.pkl` | Model Artifact | Trained XGBoost Classifier binary object used by app.py. |
| `models/ensemble_isolation_forest_model.pkl` | Model Artifact | Trained Isolation Forest anomaly detection model binary object used by app.py. |
| `models/ensemble_meta_learner.pkl` | Model Artifact | Trained Logistic Regression Meta-Learner binary object used by app.py. |
| `dataset/fraud_oracle.csv` | Dataset File | Primary training and evaluation dataset (15,420 claims, 33 columns). |
| `ipynb_files/01_Random_Forest.ipynb` | Jupyter Notebook | Data exploration, feature engineering, and training notebook for Random Forest. |
| `ipynb_files/02_XGBoost.ipynb` | Jupyter Notebook | Training and evaluation notebook for XGBoost Gradient Boosting Classifier. |
| `ipynb_files/03_Isolation_Forest.ipynb` | Jupyter Notebook | Unsupervised anomaly detection experiment & threshold tuning for Isolation Forest. |
| `ipynb_files/04_Ensemble.ipynb` | Jupyter Notebook | Full stacking ensemble pipeline: cross-validation OOF generation, meta-learner training, decision fusion, and model saving. |
| `app.py` | FastAPI Backend | Python REST API server loading model artifacts, executing real-time 104-feature preprocessing, running predictions, and serving `/api/predict`, `/api/samples`, and `/api/model-details/{name}` endpoints. |
| `index.html` | Frontend UI | Single-page responsive HTML application featuring claim form grid, preset sample loader, headline verdict card, base model cards, SPA tabs, and architecture explainer. |
| `static/style.css` | Stylesheet | Custom dark slate CSS styling system, responsive grid layout, signature model colors, and metric animations. |
| `static/script.js` | Frontend Logic | Client-side SPA tab router, form management, async API fetch calls, dynamic DOM updates, and Chart.js chart rendering. |
| `static/vendor/chart.min.js` | Vendor Library | Offline Chart.js UMD bundle providing chart visualization without internet connection. |
| `requirements.txt` | Configuration | List of Python package dependencies for running the system. |
| `README.md` | Documentation | Detailed project manual, file map, model metrics, setup instructions, and viva preparation guide. |

---

## 🤖 Model Architectures & Stacking Strategy

```
                          ┌───────────────────────────┐
                          │   Raw Claim Features      │
                          │   (31 Form Attributes)    │
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │   Preprocessing Pipeline    │
                         │ (OHE + Scale + Ordinal)     │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │ 104-Feature Encoding Vector │
                         └──────┬───────┬───────┬──────┘
                                │       │       │
       ┌────────────────────────┘       │       └────────────────────────┐
       ▼                                ▼                                ▼
┌──────────────┐                 ┌──────────────┐                 ┌──────────────┐
│Random Forest │                 │   XGBoost    │                 │  Isolation   │
│ Classifier   │                 │ Classifier   │                 │    Forest    │
└──────┬───────┘                 └──────┬───────┘                 └──────┬───────┘
       │ (Probability)                  │ (Probability)                  │ (Anomaly Score)
       └────────────────────────┐       │       ┌────────────────────────┘
                                ▼       ▼       ▼
                          ┌───────────────────────────┐
                          │ Logistic Regression Meta- │
                          │ Learner Stacking Ensemble │
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │ Final Decision Verdict &  │
                          │ 3-Tier Fusion Assessment  │
                          └───────────────────────────┘
```

### Base Models
- **Random Forest**: Constructs 250 balanced decision trees (`min_samples_leaf=2`). Excels at identifying interaction rules between policy characteristics (e.g. `Fault`, `VehicleCategory`, `Age`).
- **XGBoost (Gradient Boosting)**: Sequentially trains 150 decision trees (`max_depth=3`, `learning_rate=0.05`), using `scale_pos_weight=15.69` to maximize fraud detection recall.
- **Isolation Forest**: Builds 200 random isolation trees to partition feature space. Claims requiring fewer splits to isolate are flagged with higher anomaly scores (contamination anchored to 5.98%).

### Meta-Learner (Stacking)
- The meta-learner is a **Logistic Regression model** fit on out-of-fold predictions.
- **Meta-Learner Coefficients**: XGBoost = `4.6075`, Random Forest = `1.7913`, Isolation Forest = `0.8582`, Intercept = `-3.2246`.
- **Decision Fusion Rules**:
  - **High-confidence Fraud**: Meta-learner probability $\ge 65\%$ AND at least 2 base models flag fraud.
  - **High-confidence Legit**: Meta-learner probability $\le 35\%$ AND at most 1 base model flags fraud.
  - **Disagreement - Human Review**: Conflicting signals across models, routing the claim to human adjuster review.

---

## 📊 Empirical Model Performance Metrics (Test Set Evaluation)

Below are the static evaluation metrics computed on the test split (Stratified 20% holdout set, $N = 3,084$ claims):

| Model Name | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Confusion Matrix `[TN, FP, FN, TP]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **91.60%** | 27.44% | 24.32% | 25.79% | **0.8424** | 0.2392 | `[2780, 119, 140, 45]` |
| **XGBoost (Gradient Boosting)** | 63.85% | 13.27% | **90.81%** | 23.16% | 0.8339 | 0.2283 | `[1801, 1098, 17, 168]` |
| **Isolation Forest** | 86.93% | 7.75% | 10.81% | 9.03% | 0.4954 | 0.0649 | `[2661, 238, 165, 20]` |
| **Meta-Learner Ensemble** | 64.53% | 13.32% | **89.19%** | **23.17%** | **0.8417** | **0.2537** | `[1825, 1074, 20, 165]` |

---

## 🌐 API Endpoints

- `POST /api/predict` — Accepts claim JSON, runs preprocessing, predicts across RF, XGBoost, Isolation Forest, and Meta-Learner, returns predictions, feature importances, and decision fusion.
- `GET /api/samples` — Returns pre-configured `fraud_sample`, `legit_sample`, and `borderline_sample` claims.
- `GET /api/model-details/{model_name}` — Returns hyperparameters, search method, search space, training split details, confusion matrix, ROC/PR curve points, and feature/anomaly distribution for deep-dive tabs.

---

## 🚀 Setup & Run Instructions

### Step 1: Install Dependencies
Open PowerShell or Terminal in the project root folder and run:
```powershell
C:\Users\ASUS\anaconda3\python.exe -m pip install -r requirements.txt
```

### Step 2: Start the FastAPI Backend Server
Run Uvicorn server:
```powershell
C:\Users\ASUS\anaconda3\python.exe -m uvicorn app:app --host 127.0.0.1 --port 5000 --reload
```

### Step 3: Launch in Web Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- Click **"🚨 Load Fraud Example"** or **"✅ Load Legitimate Example"** to test preset claims.
- Use the top navigation bar to inspect deep-dive model analysis tabs (**Random Forest**, **XGBoost**, **Isolation Forest**).
