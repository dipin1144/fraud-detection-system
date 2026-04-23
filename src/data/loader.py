import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(data_dir=None):
    base = data_dir or r"C:\Users\dipin\fraud-detection-system\data\raw"

    train = pd.read_csv(os.path.join(base, "train_transaction.csv"))
    identity = pd.read_csv(os.path.join(base, "train_identity.csv"))
    df = train.merge(identity, on="TransactionID", how="left")

    X = df.drop(columns=["isFraud", "TransactionID"])
    y = df["isFraud"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
