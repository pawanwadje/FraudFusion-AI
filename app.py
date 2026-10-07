import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, Any, Optional
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve

app = FastAPI(
    title="Insurance Claim Fraud Detection API",
    description="Ensemble Learning Fraud Detection System (Random Forest, XGBoost, Isolation Forest & Stacking Meta-Learner)",
    version="2.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and preprocessors
models = {}
preprocessor = {}
static_metrics = {}
model_deep_details = {}
ISO_BOUNDS = {"min": 0.39344, "max": 0.54079}

# Column Definitions
NUMERIC_COLS = ['WeekOfMonth', 'WeekOfMonthClaimed', 'Age', 'RepNumber', 'Deductible', 'DriverRating', 'Year']
ORDINAL_MAPS = {
    'Days_Policy_Accident': ['none', '1 to 7', '8 to 15', '15 to 30', 'more than 30'],
    'Days_Policy_Claim':    ['none', '8 to 15', '15 to 30', 'more than 30'],
    'PastNumberOfClaims':   ['none', '1', '2 to 4', 'more than 4'],
    'AgeOfVehicle':         ['new', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', 'more than 7'],
    'AgeOfPolicyHolder':    ['16 to 17', '18 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 40', '41 to 50', '51 to 65', 'over 65'],
    'NumberOfSuppliments':  ['none', '1 to 2', '3 to 5', 'more than 5'],
    'AddressChange_Claim':  ['no change', 'under 6 months', '1 year', '2 to 3 years', '4 to 8 years'],
    'NumberOfCars':         ['1 vehicle', '2 vehicles', '3 to 4', '5 to 8', 'more than 8'],
    'VehiclePrice':         ['less than 20000', '20000 to 29000', '30000 to 39000', '40000 to 59000', '60000 to 69000', 'more than 69000'],
}
ORDINAL_COLS = list(ORDINAL_MAPS.keys())
CATEGORICAL_COLS = [
    'Month', 'DayOfWeek', 'Make', 'AccidentArea', 'DayOfWeekClaimed', 'MonthClaimed',
    'Sex', 'MaritalStatus', 'Fault', 'PolicyType', 'VehicleCategory',
    'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'BasePolicy'
]

def load_artifacts():
    """Load model .pkl files and initialize preprocessors and deep details from dataset."""
    global models, preprocessor, static_metrics, model_deep_details
    
    # 1. Load ML models from models/ directory
    model_paths = {
        'rf': os.path.join('models', 'ensemble_rf_model.pkl') if os.path.exists(os.path.join('models', 'ensemble_rf_model.pkl')) else 'ensemble_rf_model.pkl',
        'xgb': os.path.join('models', 'ensemble_boosting_model.pkl') if os.path.exists(os.path.join('models', 'ensemble_boosting_model.pkl')) else 'ensemble_boosting_model.pkl',
        'iso': os.path.join('models', 'ensemble_isolation_forest_model.pkl') if os.path.exists(os.path.join('models', 'ensemble_isolation_forest_model.pkl')) else 'ensemble_isolation_forest_model.pkl',
        'meta': os.path.join('models', 'ensemble_meta_learner.pkl') if os.path.exists(os.path.join('models', 'ensemble_meta_learner.pkl')) else 'ensemble_meta_learner.pkl'
    }
    
    for key, path in model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required model artifact missing: {path}")
        models[key] = joblib.load(path)
        print(f"Loaded {key} model from {path}")
        
    # 2. Fit preprocessor on dataset & evaluate curve data
    dataset_path = 'dataset/fraud_oracle.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset missing: {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    df.loc[(df['Age'] == 0) & (df['AgeOfPolicyHolder'] == '16 to 17'), 'Age'] = 17
    df.loc[df['DayOfWeekClaimed'] == '0', 'DayOfWeekClaimed'] = 'Monday'
    df.loc[df['MonthClaimed'] == '0', 'MonthClaimed'] = 'Jul'
    if 'PolicyNumber' in df.columns:
        df = df.drop(columns=['PolicyNumber'])
        
    X = df.drop(columns=['FraudFound_P'])
    y = df['FraudFound_P']

    # Ordinal Encoding
    X_ord = X.copy()
    for col, order in ORDINAL_MAPS.items():
        X_ord[col] = X_ord[col].map({v: i for i, v in enumerate(order)})
        
    # Fit OneHotEncoder & StandardScaler
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(X_ord[CATEGORICAL_COLS])
    
    scaler = StandardScaler()
    scaler.fit(X_ord[NUMERIC_COLS])
    
    preprocessor['ohe'] = ohe
    preprocessor['scaler'] = scaler
    preprocessor['ohe_cols'] = ohe.get_feature_names_out(CATEGORICAL_COLS)

    # Compute ROC, PR, and Score Distribution data on test split for model detail pages
    from sklearn.model_selection import train_test_split
    _, X_test_raw, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    X_test_ord = ordinal_encode_df(X_test_raw)
    num_test = pd.DataFrame(scaler.transform(X_test_ord[NUMERIC_COLS]), columns=NUMERIC_COLS, index=X_test_ord.index)
    ohe_test = pd.DataFrame(ohe.transform(X_test_ord[CATEGORICAL_COLS]), columns=preprocessor['ohe_cols'], index=X_test_ord.index)
    X_test_final = pd.concat([num_test, X_test_ord[ORDINAL_COLS], ohe_test], axis=1)

    # Predict test probabilities for exact curves
    rf_proba = models['rf'].predict_proba(X_test_final)[:, 1]
    xgb_proba = models['xgb'].predict_proba(X_test_final)[:, 1]
    iso_scores = -models['iso'].score_samples(X_test_final)

    # RF Curves
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, rf_proba)
    
    # XGB Curves
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, xgb_proba)

    # ISO Score Distribution
    counts_legit, bin_edges = np.histogram(iso_scores[y_test == 0], bins=15)
    counts_fraud, _ = np.histogram(iso_scores[y_test == 1], bins=bin_edges)
    bin_centers = [round((bin_edges[i] + bin_edges[i+1]) / 2, 4) for i in range(len(bin_edges)-1)]

    # RF Top Features
    rf_imp = models['rf'].feature_importances_
    rf_names = models['rf'].feature_names_in_
    rf_top_idx = np.argsort(rf_imp)[::-1][:10]
    rf_features = [{"feature": str(rf_names[i]), "importance": float(round(rf_imp[i] * 100, 2))} for i in rf_top_idx]

    # XGB Top Features
    xgb_imp = models['xgb'].feature_importances_
    xgb_names = models['xgb'].feature_names_in_
    xgb_top_idx = np.argsort(xgb_imp)[::-1][:10]
    xgb_features = [{"feature": str(xgb_names[i]), "importance": float(round(xgb_imp[i] * 100, 2))} for i in xgb_top_idx]

    # Static model metrics evaluated on test set (20% split, N=3,084)
    static_metrics = {
        "random_forest": {
            "name": "Random Forest",
            "algorithm": "Bagging Ensemble of Decision Trees",
            "accent_color": "#10B981",
            "accuracy": 0.9160,
            "precision": 0.2744,
            "recall": 0.2432,
            "f1": 0.2579,
            "roc_auc": 0.8424,
            "pr_auc": 0.2392,
            "confusion_matrix": {"tn": 2780, "fp": 119, "fn": 140, "tp": 45}
        },
        "xgboost": {
            "name": "XGBoost",
            "algorithm": "Sequential Gradient Boosted Decision Trees",
            "accent_color": "#0EA5E9",
            "accuracy": 0.6385,
            "precision": 0.1327,
            "recall": 0.9081,
            "f1": 0.2316,
            "roc_auc": 0.8339,
            "pr_auc": 0.2283,
            "confusion_matrix": {"tn": 1801, "fp": 1098, "fn": 17, "tp": 168}
        },
        "isolation_forest": {
            "name": "Isolation Forest",
            "algorithm": "Unsupervised Anomaly Detection",
            "accent_color": "#F59E0B",
            "accuracy": 0.8693,
            "precision": 0.0775,
            "recall": 0.1081,
            "f1": 0.0903,
            "roc_auc": 0.4954,
            "pr_auc": 0.0649,
            "confusion_matrix": {"tn": 2661, "fp": 238, "fn": 165, "tp": 20}
        },
        "ensemble": {
            "name": "Meta-Learner Stacking Ensemble",
            "accent_color": "#F43F5E",
            "algorithm": "Logistic Regression Meta-Classifier",
            "accuracy": 0.6453,
            "precision": 0.1332,
            "recall": 0.8919,
            "f1": 0.2317,
            "roc_auc": 0.8417,
            "pr_auc": 0.2537,
            "confusion_matrix": {"tn": 1825, "fp": 1074, "fn": 20, "tp": 165}
        }
    }

    # Model Deep Details Dictionary
    model_deep_details = {
        "random_forest": {
            **static_metrics["random_forest"],
            "role": "Captures non-linear decision boundaries and feature interaction rules across policyholder attributes.",
            "hyperparameters": {
                "n_estimators": 250,
                "min_samples_leaf": 2,
                "max_depth": "None (unlimited)",
                "class_weight": "balanced",
                "random_state": 42
            },
            "tuning_details": {
                "search_method": "RandomizedSearchCV",
                "cv_folds": "Stratified 3-Fold CV",
                "scoring_metric": "PR-AUC (Average Precision)",
                "search_space": {
                    "n_estimators": "[150, 250]",
                    "max_depth": "[10, None]",
                    "min_samples_leaf": "[1, 2]"
                },
                "best_params": "{'n_estimators': 250, 'min_samples_leaf': 2, 'max_depth': None}"
            },
            "training_details": {
                "total_dataset_size": "15,420 claims",
                "train_test_split": "80% Train (12,336 claims) / 20% Test (3,084 claims)",
                "encoded_features": "104 features",
                "class_imbalance_handling": "Cost-sensitive reweighting (class_weight='balanced')"
            },
            "top_features": rf_features,
            "roc_curve": {"fpr": sample_coords(fpr_rf), "tpr": sample_coords(tpr_rf)},
            "pr_curve": {"precision": sample_coords(prec_rf), "recall": sample_coords(rec_rf)}
        },
        "xgboost": {
            **static_metrics["xgboost"],
            "role": "Sequentially optimizes residual errors to achieve high fraud recall (89.7%) on hard edge-case claims.",
            "hyperparameters": {
                "n_estimators": 150,
                "max_depth": 3,
                "learning_rate": 0.05,
                "scale_pos_weight": 15.69,
                "eval_metric": "aucpr",
                "random_state": 42
            },
            "tuning_details": {
                "search_method": "RandomizedSearchCV",
                "cv_folds": "Stratified 3-Fold CV",
                "scoring_metric": "PR-AUC (Average Precision)",
                "search_space": {
                    "n_estimators": "[150, 250]",
                    "max_depth": "[3, 5]",
                    "learning_rate": "[0.05, 0.1]"
                },
                "best_params": "{'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.05}"
            },
            "training_details": {
                "total_dataset_size": "15,420 claims",
                "train_test_split": "80% Train (12,336 claims) / 20% Test (3,084 claims)",
                "encoded_features": "104 features",
                "class_imbalance_handling": "scale_pos_weight=15.69 (Ratio of 11,597 legit / 739 fraud train samples)"
            },
            "top_features": xgb_features,
            "roc_curve": {"fpr": sample_coords(fpr_xgb), "tpr": sample_coords(tpr_xgb)},
            "pr_curve": {"precision": sample_coords(prec_xgb), "recall": sample_coords(rec_xgb)}
        },
        "isolation_forest": {
            **static_metrics["isolation_forest"],
            "role": "Unsupervised anomaly detection model that isolates unusual claim profiles without relying on class labels.",
            "hyperparameters": {
                "n_estimators": 200,
                "contamination": 0.0598,
                "max_samples": "auto",
                "bootstrap": False,
                "random_state": 42
            },
            "tuning_details": {
                "search_method": "Contamination Anchor",
                "cv_folds": "Stratified 3-Fold OOF Scoring",
                "scoring_metric": "Out-of-Fold Score Samples",
                "search_space": "Anchored to true training fraud ratio (739 / 12336 = 0.0598)",
                "best_params": "{'n_estimators': 200, 'contamination': 0.0598}"
            },
            "training_details": {
                "total_dataset_size": "15,420 claims",
                "train_test_split": "80% Train (12,336 claims) / 20% Test (3,084 claims)",
                "encoded_features": "104 features",
                "class_imbalance_handling": "Unsupervised spatial isolation density thresholding"
            },
            "score_distribution": {
                "bin_centers": bin_centers,
                "legit_counts": counts_legit.tolist(),
                "fraud_counts": counts_fraud.tolist(),
                "threshold": 0.467
            }
        }
    }
    print("Artifacts, preprocessors, and model details successfully loaded.")

def ordinal_encode_df(data):
    data = data.copy()
    for col, order in ORDINAL_MAPS.items():
        data[col] = data[col].map({v: i for i, v in enumerate(order)})
    return data

def sample_coords(arr, n=30):
    """Sample array to n clean data points for light JSON payload."""
    indices = np.linspace(0, len(arr) - 1, n, dtype=int)
    return [float(round(arr[i], 4)) for i in indices]

@app.on_event("startup")
def startup_event():
    load_artifacts()

def preprocess_claim(raw_claim: Dict[str, Any]) -> pd.DataFrame:
    """Transform raw JSON claim dict into 104 encoded feature vector."""
    df_single = pd.DataFrame([raw_claim])
    
    # Cleaning
    if 'Age' in df_single.columns and 'AgeOfPolicyHolder' in df_single.columns:
        if (df_single['Age'].iloc[0] == 0 or pd.isna(df_single['Age'].iloc[0])) and df_single['AgeOfPolicyHolder'].iloc[0] == '16 to 17':
            df_single['Age'] = 17
            
    if 'DayOfWeekClaimed' in df_single.columns and str(df_single['DayOfWeekClaimed'].iloc[0]) == '0':
        df_single['DayOfWeekClaimed'] = 'Monday'
        
    if 'MonthClaimed' in df_single.columns and str(df_single['MonthClaimed'].iloc[0]) == '0':
        df_single['MonthClaimed'] = 'Jul'
        
    if 'PolicyNumber' in df_single.columns:
        df_single = df_single.drop(columns=['PolicyNumber'])
        
    # Ordinal encoding
    df_ord = df_single.copy()
    for col, order in ORDINAL_MAPS.items():
        val = df_ord[col].iloc[0] if col in df_ord.columns else order[0]
        idx = order.index(val) if val in order else 0
        df_ord[col] = idx
        
    # Scaling & One-Hot Encoding
    scaler = preprocessor['scaler']
    ohe = preprocessor['ohe']
    
    for col in NUMERIC_COLS:
        df_ord[col] = pd.to_numeric(df_ord[col], errors='coerce').fillna(0)
        
    num_encoded = pd.DataFrame(
        scaler.transform(df_ord[NUMERIC_COLS]),
        columns=NUMERIC_COLS,
        index=df_single.index
    )
    
    ohe_encoded = pd.DataFrame(
        ohe.transform(df_ord[CATEGORICAL_COLS]),
        columns=preprocessor['ohe_cols'],
        index=df_single.index
    )
    
    final_df = pd.concat([num_encoded, df_ord[ORDINAL_COLS], ohe_encoded], axis=1)
    
    rf_cols = models['rf'].feature_names_in_
    final_df = final_df.reindex(columns=rf_cols, fill_value=0)
    return final_df

@app.post("/api/predict")
def predict_claim(claim: Dict[str, Any]):
    """Accept claim details, execute base models & meta-learner, return predictions."""
    try:
        encoded_df = preprocess_claim(claim)
        
        # 1. Random Forest
        rf_model = models['rf']
        rf_prob = float(rf_model.predict_proba(encoded_df)[:, 1][0])
        rf_label = "Fraud" if rf_prob >= 0.5 else "Not Fraud"
        
        # 2. XGBoost
        xgb_model = models['xgb']
        xgb_prob = float(xgb_model.predict_proba(encoded_df)[:, 1][0])
        xgb_label = "Fraud" if xgb_prob >= 0.5 else "Not Fraud"
        
        # 3. Isolation Forest
        iso_model = models['iso']
        iso_raw_score = float(-iso_model.score_samples(encoded_df)[0])
        iso_min, iso_max = ISO_BOUNDS["min"], ISO_BOUNDS["max"]
        iso_norm = float(np.clip((iso_raw_score - iso_min) / (iso_max - iso_min), 0.0, 1.0))
        iso_label = "Fraud" if iso_norm >= 0.5 else "Not Fraud"
        
        # 4. Meta-Learner Stacking Ensemble
        meta_model = models['meta']
        meta_input = np.array([[rf_prob, xgb_prob, iso_raw_score]])
        meta_prob = float(meta_model.predict_proba(meta_input)[:, 1][0])
        meta_label = "Fraud" if meta_prob >= 0.5 else "Not Fraud"
        
        # Decision Fusion Logic
        rf_flag = int(rf_prob >= 0.3)
        boost_flag = int(xgb_prob >= 0.3)
        if_flag = int(iso_norm >= 0.5)
        votes = rf_flag + boost_flag + if_flag
        
        if meta_prob >= 0.65 and votes >= 2:
            fusion_decision = "High-confidence Fraud"
            fusion_badge = "fraud"
            explanation = "High risk confirmed: The meta-learner recorded strong fraud probability (≥65%) with consensus from at least 2 base models."
        elif meta_prob <= 0.35 and votes <= 1:
            fusion_decision = "High-confidence Legit"
            fusion_badge = "legit"
            explanation = "Low risk confirmed: Meta-learner registered low fraud risk (≤35%) with unanimous or majority legitimate consensus."
        else:
            fusion_decision = "Disagreement - Human Review"
            fusion_badge = "review"
            explanation = "Moderate / Disagreeing risk signal: The base models show conflicting indicators. Claim is flagged for human adjuster review."
            
        base_labels = [rf_label, xgb_label, iso_label]
        all_agree = (len(set(base_labels)) == 1)
        if all_agree:
            agreement_note = f"All 3 base models unanimously agree on a '{rf_label}' verdict."
        else:
            agreement_note = f"Base models show split indicators ({base_labels.count('Fraud')} Fraud vs {base_labels.count('Not Fraud')} Legitimate)."

        rf_importances = rf_model.feature_importances_
        rf_feat_names = rf_model.feature_names_in_
        top_indices = np.argsort(rf_importances)[::-1][:8]
        top_features = [
            {"feature": str(rf_feat_names[idx]), "importance": float(round(rf_importances[idx] * 100, 2))}
            for idx in top_indices
        ]

        return {
            "status": "success",
            "claim_summary": {
                "make": claim.get("Make", "Unknown"),
                "policy_type": claim.get("PolicyType", "Unknown"),
                "age": claim.get("Age", 0),
                "fault": claim.get("Fault", "Unknown"),
                "deductible": claim.get("Deductible", 0)
            },
            "random_forest": {
                "name": "Random Forest",
                "algorithm": "Bagging Ensemble",
                "label": rf_label,
                "confidence": float(round(rf_prob * 100, 1)),
                "raw_prob": float(round(rf_prob, 4)),
                "metrics": static_metrics["random_forest"]
            },
            "xgboost": {
                "name": "XGBoost",
                "algorithm": "Gradient Boosting",
                "label": xgb_label,
                "confidence": float(round(xgb_prob * 100, 1)),
                "raw_prob": float(round(xgb_prob, 4)),
                "metrics": static_metrics["xgboost"]
            },
            "isolation_forest": {
                "name": "Isolation Forest",
                "algorithm": "Anomaly Detection",
                "label": iso_label,
                "confidence": float(round(iso_norm * 100, 1)),
                "raw_prob": float(round(iso_norm, 4)),
                "raw_anomaly_score": float(round(iso_raw_score, 4)),
                "metrics": static_metrics["isolation_forest"]
            },
            "ensemble": {
                "name": "Meta-Learner Stacking Ensemble",
                "algorithm": "Logistic Regression Stacking",
                "label": meta_label,
                "confidence": float(round(meta_prob * 100, 1)),
                "raw_prob": float(round(meta_prob, 4)),
                "decision_fusion": fusion_decision,
                "fusion_badge": fusion_badge,
                "explanation": explanation,
                "metrics": static_metrics["ensemble"]
            },
            "comparison": {
                "all_agree": all_agree,
                "agreement_note": agreement_note,
                "votes_fraud": votes,
                "votes_total": 3
            },
            "top_features": top_features
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/samples")
def get_sample_claims():
    """Return realistic ready-made fraud, legitimate, and borderline claim samples."""
    sample_fraud = {
        "Month": "Jul",
        "WeekOfMonth": 1,
        "DayOfWeek": "Saturday",
        "Make": "Honda",
        "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Tuesday",
        "MonthClaimed": "Sep",
        "WeekOfMonthClaimed": 4,
        "Sex": "Male",
        "MaritalStatus": "Single",
        "Age": 17,
        "Fault": "Policy Holder",
        "PolicyType": "Sedan - All Perils",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "more than 69000",
        "RepNumber": 9,
        "Deductible": 400,
        "DriverRating": 1,
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "PastNumberOfClaims": "none",
        "AgeOfVehicle": "new",
        "AgeOfPolicyHolder": "16 to 17",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "AgentType": "External",
        "NumberOfSuppliments": "none",
        "AddressChange_Claim": "no change",
        "NumberOfCars": "1 vehicle",
        "Year": 1994,
        "BasePolicy": "All Perils"
    }
    
    sample_legit = {
        "Month": "Dec",
        "WeekOfMonth": 5,
        "DayOfWeek": "Wednesday",
        "Make": "Honda",
        "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Tuesday",
        "MonthClaimed": "Jan",
        "WeekOfMonthClaimed": 1,
        "Sex": "Female",
        "MaritalStatus": "Single",
        "Age": 21,
        "Fault": "Third Party",
        "PolicyType": "Sport - Liability",
        "VehicleCategory": "Sport",
        "VehiclePrice": "more than 69000",
        "RepNumber": 12,
        "Deductible": 300,
        "DriverRating": 1,
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "PastNumberOfClaims": "none",
        "AgeOfVehicle": "3 years",
        "AgeOfPolicyHolder": "26 to 30",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "AgentType": "External",
        "NumberOfSuppliments": "none",
        "AddressChange_Claim": "1 year",
        "NumberOfCars": "3 to 4",
        "Year": 1994,
        "BasePolicy": "Liability"
    }

    sample_borderline = {
        "Month": "Jan",
        "WeekOfMonth": 2,
        "DayOfWeek": "Monday",
        "Make": "Toyota",
        "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Wednesday",
        "MonthClaimed": "Jan",
        "WeekOfMonthClaimed": 2,
        "Sex": "Male",
        "MaritalStatus": "Married",
        "Age": 35,
        "Fault": "Policy Holder",
        "PolicyType": "Sedan - Collision",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "30000 to 39000",
        "RepNumber": 7,
        "Deductible": 400,
        "DriverRating": 3,
        "Days_Policy_Accident": "15 to 30",
        "Days_Policy_Claim": "8 to 15",
        "PastNumberOfClaims": "2 to 4",
        "AgeOfVehicle": "5 years",
        "AgeOfPolicyHolder": "31 to 35",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "AgentType": "External",
        "NumberOfSuppliments": "1 to 2",
        "AddressChange_Claim": "under 6 months",
        "NumberOfCars": "1 vehicle",
        "Year": 1995,
        "BasePolicy": "Collision"
    }

    return {
        "fraud_sample": sample_fraud,
        "legit_sample": sample_legit,
        "borderline_sample": sample_borderline
    }

@app.get("/api/model-details/{model_name}")
def get_model_details(model_name: str):
    """Return comprehensive hyperparameters, training details, metrics, and curve data for a specific model."""
    clean_key = model_name.lower().replace("-", "_").replace(" ", "_")
    if clean_key not in model_deep_details:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Valid options: random_forest, xgboost, isolation_forest")
    return model_deep_details[clean_key]

# Serve static files & root routes explicitly
@app.get("/style.css")
def get_style_css():
    return FileResponse(os.path.join("static", "style.css"), media_type="text/css")

@app.get("/script.js")
def get_script_js():
    return FileResponse(os.path.join("static", "script.js"), media_type="text/javascript")

if os.path.exists("index.html"):
    @app.get("/")
    def read_root():
        return FileResponse("index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
