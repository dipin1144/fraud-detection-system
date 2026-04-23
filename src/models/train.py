import pandas as pd
import numpy as np
import joblib
import logging
import yaml
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, 
                              precision_score, recall_score,
                              classification_report)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_feature_columns(df, target='isFraud'):
    """Get feature columns excluding target and ID columns."""
    exclude = [target, 'TransactionID', 'TransactionDT']
    cols = [c for c in df.columns if c not in exclude]
    return cols


def build_baseline_pipeline():
    """Logistic Regression baseline pipeline."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])


def build_xgb_pipeline(scale_pos_weight=28):
    """XGBoost pipeline."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            verbosity=0
        ))
    ])


def evaluate_model(model, X, y, name="Model"):
    """Evaluate model and print metrics."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y, y_pred_proba)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    logger.info(f"\n{name} Results:")
    logger.info(f"  AUC-ROC:   {auc:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")

    return {
        'auc': auc, 'f1': f1,
        'precision': precision, 'recall': recall
    }


def save_model(model, path):
    """Save model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def train_baseline(X_train, y_train, X_val, y_val):
    """Train and evaluate baseline logistic regression."""
    logger.info("Training baseline Logistic Regression...")
    pipeline = build_baseline_pipeline()
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_val, y_val, "Baseline LR")
    return pipeline, metrics


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train and evaluate XGBoost model."""
    scale_pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())
    logger.info(f"Training XGBoost (scale_pos_weight={scale_pos_weight})...")
    pipeline = build_xgb_pipeline(scale_pos_weight)
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_val, y_val, "XGBoost")
    return pipeline, metrics