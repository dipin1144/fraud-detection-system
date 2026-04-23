# Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.13-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-green) ![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688) ![AUC](https://img.shields.io/badge/AUC--ROC-0.9566-brightgreen)

A production-grade fraud detection system trained on the IEEE-CIS dataset (590K transactions), featuring a tuned XGBoost model, REST API, and real-time Streamlit dashboard.

---

## Results

| Model | AUC-ROC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.8424 | 0.1872 | 0.1067 | 0.7597 |
| XGBoost (baseline) | 0.9268 | 0.3774 | 0.2490 | 0.7787 |
| **XGBoost (Optuna tuned)** | **0.9566** | **0.7020** | **0.7803** | **0.6381** |

> Optuna tuning over 50 trials improved AUC by +0.0298 and Precision by +0.4759

---

## Key Findings

- 28:1 class imbalance handled via scale_pos_weight
- Peak fraud activity at 7AM
- Product C has highest fraud rate
- Top fraud signals: addr1_count, P_emaildomain, card velocity features
- Optimal classification threshold: 0.5905

---

## Tech Stack

- **ML**: XGBoost, Scikit-learn, Optuna, SHAP, Imbalanced-learn
- **API**: FastAPI, Uvicorn, Pydantic
- **Dashboard**: Streamlit
- **Data**: IEEE-CIS Fraud Detection Dataset (Kaggle)
- **Language**: Python 3.13

---

## Model Explainability

SHAP analysis reveals the top fraud signals:
- `addr1_count` - high transaction velocity from same address
- `P_emaildomain` - certain email domains correlate with fraud
- `hour` - time of day (peak fraud at 7AM)
- `card6` - card category is a strong predictor

---

## Quick Start

1. Install: `pip install fastapi uvicorn streamlit xgboost scikit-learn shap optuna joblib pandas numpy`
2. Start API: `uvicorn src.api.main:app --reload --port 8000`
3. Start Dashboard: `streamlit run dashboard.py`
4. API Docs: http://127.0.0.1:8000/docs

---

*Built by Dipin | Targeting: Revolut, Adyen, N26, Wise*
