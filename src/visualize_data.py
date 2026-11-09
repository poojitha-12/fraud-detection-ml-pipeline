
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the generated dataset
df = pd.read_csv("data/transactions.csv")

# Count legitimate and fraudulent transactions
counts = df["is_fraud"].value_counts()
legitimate = counts.get(0, 0)
fraud = counts.get(1, 0)

# Create the chart
plt.figure(figsize=(8, 5))

bars = plt.bar(
    ["Legitimate", "Fraud"],
    [legitimate, fraud],
    color=["steelblue", "tomato"]
)

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

# Display counts above the bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

# Save chart
Path("results").mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig("results/fraud_distribution.png", dpi=150)
plt.close()

print("Visualization created successfully!")
print("Legitimate transactions:", legitimate)
print("Fraudulent transactions:", fraud)
print("Chart saved to results/fraud_distribution.png")
