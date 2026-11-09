
# Fraud Detection ML Pipeline

An end-to-end machine learning project for detecting
fraudulent financial transactions.

## Project Overview

This project demonstrates how to build a fraud detection
pipeline using Python and machine learning.

## Day 1: Data Generation and Visualization

Generated 1,000 synthetic financial transactions.

### Dataset Results

- Total transactions: 1,000
- Legitimate transactions: 956 (95.6%)
- Fraudulent transactions: 44 (4.4%)

### Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (planned for model training)

### Project Structure

- `src/generate_data.py` - Generates synthetic data
- `src/visualize_data.py` - Creates fraud visualization
- `data/transactions.csv` - Synthetic dataset
- `results/fraud_distribution.png` - Fraud distribution chart

### Visualization

![Fraud Distribution](results/fraud_distribution.png)

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate transactions:

```bash
python src/generate_data.py
```

Create visualization:

```bash
python src/visualize_data.py
```

## Next Steps

- Exploratory data analysis
- Feature engineering
- Train machine learning models
- Evaluate precision, recall, and F1-score
- Build an inference pipeline
