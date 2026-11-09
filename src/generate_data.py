
import numpy as np
import pandas as pd
from pathlib import Path

# Generate 1,000 synthetic transactions
rng = np.random.default_rng(42)
n = 1000

amount = rng.lognormal(3.5, 1.0, n)
hour = rng.integers(0, 24, n)
international = rng.binomial(1, 0.15, n)

# Create simulated fraud probabilities
risk = (
    -4.0
    + 0.008 * amount
    + 0.8 * international
    + 0.5 * (hour < 5)
)

probability = 1 / (1 + np.exp(-risk))
is_fraud = rng.binomial(1, probability)

# Create the dataset
df = pd.DataFrame({
    "transaction_id": range(1, n + 1),
    "amount": np.round(amount, 2),
    "hour": hour,
    "international": international,
    "is_fraud": is_fraud
})

# Save transactions to CSV
Path("data").mkdir(exist_ok=True)
df.to_csv("data/transactions.csv", index=False)

print("Dataset generated successfully!")
print("Total transactions:", len(df))

print("\nFraud distribution:")
print(df["is_fraud"].value_counts())

print("\nFirst five transactions:")
print(df.head())
