
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ------------------------------------
# STEP 1: Load prepared datasets
# ------------------------------------

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

print("=== DAY 4: FRAUD DETECTION MODEL ===")

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))

# ------------------------------------
# STEP 2: Separate features and labels
# ------------------------------------

X_train = train_df.drop(columns=["is_fraud"])
y_train = train_df["is_fraud"]

X_test = test_df.drop(columns=["is_fraud"])
y_test = test_df["is_fraud"]

print("\nFeatures used:")
print(list(X_train.columns))

# ------------------------------------
# STEP 3: Train Logistic Regression
# ------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed!")

# ------------------------------------
# STEP 4: Generate predictions
# ------------------------------------

y_pred = model.predict(X_test)

# Fraud probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# ------------------------------------
# STEP 5: Evaluate model
# ------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\n=== MODEL PERFORMANCE ===")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ------------------------------------
# STEP 6: Analyze confusion matrix
# ------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\n=== CONFUSION MATRIX ===")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# ------------------------------------
# STEP 7: Save visualization
# ------------------------------------

Path("results").mkdir(exist_ok=True)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)

display.plot(cmap="Blues", values_format="d")

plt.title("Day 4: Logistic Regression Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=150
)

plt.close()

# ------------------------------------
# STEP 8: Save trained model
# ------------------------------------

Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/logistic_regression.joblib"
)

# ------------------------------------
# STEP 9: Save performance metrics
# ------------------------------------

metrics_df = pd.DataFrame([{
    "model": "Logistic Regression",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "true_negatives": tn,
    "false_positives": fp,
    "false_negatives": fn,
    "true_positives": tp
}])

metrics_df.to_csv(
    "results/model_metrics.csv",
    index=False
)

print("\nDay 4 completed successfully!")
print("Model saved to models/logistic_regression.joblib")
print("Confusion matrix saved to results/confusion_matrix.png")
print("Metrics saved to results/model_metrics.csv")
