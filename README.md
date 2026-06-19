# Bank Fraud Detection System

A machine learning project that detects fraudulent bank transactions using Python and XGBoost. Built this to get hands-on experience with real-world ML problems like class imbalance and feature engineering, following along with tutorials and then expanding on it with my own experiments.

## What it does

The model looks at transaction data and predicts whether a transaction is fraudulent or not. The dataset has over 6 million transactions, but only a tiny fraction are actually fraud cases — so a big part of this project was figuring out how to handle that imbalance properly instead of just getting a high accuracy score that means nothing.

## Dataset

This project uses a bank transaction dataset from [Kaggle](https://www.kaggle.com) (~6M+ rows). The dataset file isn't included in this repo due to its size (470MB+), but the full notebook shows the exact preprocessing steps applied to it.

## Approach

- **Feature engineering** — created new features like balance differences before/after transactions, log-transformed transaction amounts, and time-of-day patterns
- **Handling class imbalance** — since fraud cases are rare, used techniques like `scale_pos_weight` and proper scaling so the model doesn't just ignore the minority class
- **Model comparison** — trained and compared Logistic Regression, Random Forest, and XGBoost to see which performed best
- **Threshold tuning** — instead of using the default 0.5 cutoff, tuned the decision threshold to prioritize catching more fraud cases (recall ≥ 90%)
- **Deployment-ready** — saved the final trained model using `joblib` so it can be reused without retraining

## Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn (for visualizing patterns and results)

## Results

The final XGBoost model was able to catch the large majority of fraudulent transactions while keeping false positives manageable — a good balance for a real banking use case where missing fraud is far more costly than a few extra alerts.

## What I learned

This was my first real deep-dive into handling imbalanced datasets and understanding why accuracy isn't a good metric when one class is rare. I also got more comfortable with the practical side of ML — going from raw data to a model you could actually deploy.

## Notes

I built this project by following along with tutorials and then experimenting further on my own to understand the "why" behind each step, not just copy the code.

---
**Author:** Muhammad Talha Ansari
**Connect:** [LinkedIn](https://www.linkedin.com/in/talha-ansari-504312375/) · [GitHub](https://github.com/M-TalhaAnsari)
