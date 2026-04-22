import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(config: dict) -> tuple:
    """Load raw transaction and identity data."""
    transaction_path = config['data']['raw_path']
    identity_path = transaction_path.replace('train_transaction', 'train_identity')

    logger.info(f"Loading transaction data from {transaction_path}")
    df_transaction = pd.read_csv(transaction_path)

    logger.info(f"Loading identity data from {identity_path}")
    df_identity = pd.read_csv(identity_path)

    logger.info(f"Transaction shape: {df_transaction.shape}")
    logger.info(f"Identity shape: {df_identity.shape}")

    return df_transaction, df_identity


def split_data(df: pd.DataFrame, config: dict) -> tuple:
    """Split data into train, validation and test sets."""
    target = config['data']['target_column']
    test_size = config['data']['test_size']
    val_size = config['data']['val_size']
    random_seed = config['data']['random_seed']

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, 
        random_state=random_seed, 
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,