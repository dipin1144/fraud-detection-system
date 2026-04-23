import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def merge_datasets(df_transaction, df_identity):
    df = df_transaction.merge(df_identity, on='TransactionID', how='left')
    return df


def add_time_features(df):
    df = df.copy()
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 6)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df


def add_amount_features(df):
    df = df.copy()
    df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
    df['TransactionAmt_decimal'] = df['TransactionAmt'] % 1
    df['is_round_amount'] = (df['TransactionAmt_decimal'] == 0).astype(int)
    return df


def add_velocity_features(df):
    df = df.copy()
    df['card1_count'] = df.groupby('card1')['card1'].transform('count')
    df['card1_amt_mean'] = df.groupby('card1')['TransactionAmt'].transform('mean')
    df['addr1_count'] = df.groupby('addr1')['addr1'].transform('count')
    return df


def drop_high_missing_features(df, threshold=0.5):
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    cols_to_drop = [c for c in cols_to_drop if c != 'isFraud']
    df = df.drop(columns=cols_to_drop)
    return df


def encode_categoricals(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes
    return df


def run_feature_engineering(df_transaction, df_identity):
    df = merge_datasets(df_transaction, df_identity)
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_velocity_features(df)
    df = drop_high_missing_features(df, threshold=0.5)
    df = encode_categoricals(df)
    return df

def engineer_features(X_train, X_val, X_test):
    """Wrapper to apply feature engineering on pre-split dataframes."""
    import pandas as pd

    def process(df):
        df = df.copy()
        df = add_time_features(df)
        df = add_amount_features(df)
        df = add_velocity_features(df)
        df = drop_high_missing_features(df, threshold=0.5)
        df = encode_categoricals(df)
        # Align columns to train
        return df

    X_train_fe = process(X_train)
    X_val_fe   = process(X_val)
    X_test_fe  = process(X_test)

    # Align val/test columns to train (in case drop_high_missing differs)
    X_val_fe  = X_val_fe.reindex(columns=X_train_fe.columns, fill_value=0)
    X_test_fe = X_test_fe.reindex(columns=X_train_fe.columns, fill_value=0)

    return X_train_fe, X_val_fe, X_test_fe
