
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Create results directory
Path("results").mkdir(exist_ok=True)

# 1. Load dataset
df = pd.read_csv("data/transactions.csv")

print("=== DAY 2: EXPLORATORY DATA ANALYSIS ===")

# 2. Basic dataset information
print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn data types:")
print(df.dtypes)

# 3. Data quality checks
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate transaction IDs:")
print(df["transaction_id"].duplicated().sum())

# 4. Descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# 5. Compare transaction amounts
print("\nTransaction amounts by fraud class:")
print(df.groupby("is_fraud")["amount"].agg(
    ["count", "mean", "median", "min", "max"]
))

plt.figure(figsize=(8, 5))

df.boxplot(column="amount", by="is_fraud")

plt.title("Transaction Amount by Fraud Class")
plt.suptitle("")
plt.xlabel("Fraud Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Transaction Amount")
plt.tight_layout()

plt.savefig(
    "results/amount_by_fraud.png",
    dpi=150
)
plt.close()

# 6. Analyze transaction hours
hourly = df.groupby("hour")["is_fraud"].agg(
    ["count", "sum", "mean"]
)

print("\nFraud by transaction hour:")
print(hourly)

plt.figure(figsize=(10, 5))
plt.plot(
    hourly.index,
    hourly["mean"] * 100,
    marker="o"
)

plt.title("Fraud Rate by Transaction Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Fraud Rate (%)")
plt.xticks(range(0, 24))
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/fraud_by_hour.png",
    dpi=150
)
plt.close()

# 7. Analyze international transactions
international = df.groupby("international")[
    "is_fraud"
].agg(["count", "sum", "mean"])

print("\nFraud by international status:")
print(international)

plt.figure(figsize=(8, 5))

plt.bar(
    ["Domestic", "International"],
    international.reindex([0, 1], fill_value=0)["mean"] * 100,
    color=["steelblue", "tomato"]
)

plt.title("Fraud Rate: Domestic vs International")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Rate (%)")
plt.tight_layout()

plt.savefig(
    "results/fraud_by_international.png",
    dpi=150
)
plt.close()

# 8. Save statistical summary
df.describe().to_csv("results/summary_statistics.csv")

print("\nEDA completed successfully!")
print("Charts and statistics saved in results/")
