
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load the dataset
df = pd.read_csv("data/transactions.csv")

print("=== DAY 3: FEATURE ENGINEERING ===")

print("\nOriginal dataset shape:", df.shape)

# 2. Create new features
df["is_night"] = (df["hour"] < 5).astype(int)
df["log_amount"] = np.log1p(df["amount"])

# 3. Select model features
feature_columns = [
    "amount",
    "hour",
    "international",
    "is_night",
    "log_amount"
]

X = df[feature_columns]
y = df["is_fraud"]

print("\nSelected features:")
print(feature_columns)

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Scale numerical features
scaler = StandardScaler()

numeric_columns = ["amount", "hour", "log_amount"]

X_train = X_train.copy()
X_test = X_test.copy()

X_train[numeric_columns] = scaler.fit_transform(
    X_train[numeric_columns]
)

X_test[numeric_columns] = scaler.transform(
    X_test[numeric_columns]
)

# 6. Save prepared datasets
Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

train_df = X_train.copy()
train_df["is_fraud"] = y_train

test_df = X_test.copy()
test_df["is_fraud"] = y_test

train_df.to_csv(
    "data/processed/train.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False
)

# 7. Save scaler for future predictions
Path("models").mkdir(exist_ok=True)

joblib.dump(
    scaler,
    "models/scaler.joblib"
)

print("\nTraining fraud distribution:")
print(y_train.value_counts())

print("\nTesting fraud distribution:")
print(y_test.value_counts())

print("\nFeature engineering completed!")
print("Files saved successfully.")
