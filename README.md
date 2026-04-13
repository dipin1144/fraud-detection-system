# Fraud Detection System

Real-time fraud detection API built with XGBoost, FastAPI, PostgreSQL,
and SHAP explainability. Trained on IEEE-CIS transaction data (590K rows).

## Live demo
[API docs](https://your-app.onrender.com/docs) | [Streamlit UI](https://your-app.streamlit.app)

## Model performance
| Metric | Score |
|--------|-------|
| AUC-ROC | TBD |
| F1 (fraud class) | TBD |
| Precision | TBD |
| Recall | TBD |

## Tech stack
- ML: XGBoost, scikit-learn, SHAP, Optuna
- API: FastAPI, Pydantic, async SQLAlchemy
- Database: PostgreSQL (predictions + audit log)
- Infra: Docker, GitHub Actions CI/CD
- Monitoring: Prometheus, Grafana, Evidently AI

## Quick start
git clone https://github.com/dipin1144/fraud-detection-system
cd fraud-detection-system
make install
make train
make api

## Project structure
See /src for modules, /notebooks for EDA