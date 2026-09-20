![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-model-EB0028)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

# Bank Fraud Detection System

A machine learning pipeline that flags fraudulent bank transactions in a
6M+ row dataset, built around the core challenge of real-world fraud
detection: **the fraud you care about is under 1% of the data.**

## At a glance

| | |
|---|---|
| **Dataset** | 6M+ bank transactions (Kaggle) |
| **Core challenge** | Severe class imbalance — fraud is a tiny fraction of transactions |
| **Models compared** | Logistic Regression, Random Forest, XGBoost |
| **Final model** | XGBoost, threshold-tuned for recall ≥ 90% |
| **Why recall-first** | In fraud detection, missing real fraud is far costlier than a false alarm |

---

## What it does

The model looks at transaction-level data and predicts whether a transaction
is fraudulent. With fraud cases this rare, a model that just predicts "not
fraud" every time would score ~99% accuracy while catching nothing — so a
core part of this project is building a pipeline that doesn't fall into that
trap.

## Approach

- **Feature engineering** — balance differences before/after transactions,
  log-transformed transaction amounts, time-of-day patterns
- **Class imbalance handling** — `scale_pos_weight` and proper scaling so
  the model doesn't collapse toward the majority class
- **Model comparison** — Logistic Regression, Random Forest, and XGBoost
  evaluated head-to-head
- **Threshold tuning** — moved off the default 0.5 cutoff and tuned the
  decision threshold to prioritize catching fraud (recall ≥ 90%)
- **Deployment-ready artifact** — final model persisted with `joblib` so it
  can be reused without retraining

## Tech stack

Python · Pandas / NumPy · Scikit-learn · XGBoost · Matplotlib / Seaborn

## Dataset

Bank transaction dataset from [Kaggle](https://www.kaggle.com) (~6M rows,
470MB+). Not included in this repo due to size — the notebook documents the
exact preprocessing steps applied to it end to end.

## Results

The final XGBoost model catches the large majority of fraudulent
transactions while keeping false positives at a manageable rate — the right
tradeoff for a banking use case, where a missed fraud case is far more
expensive than an extra manual review.

## What this project demonstrates

Handling severely imbalanced classification correctly (not just optimizing
for accuracy), comparing models rigorously rather than picking the first one
that runs, and tuning a decision threshold against the metric that actually
matters for the use case rather than the default.

---

**Author:** Muhammad Talha Ansari — [LinkedIn](https://www.linkedin.com/in/talha-ansari-504312375/) · [GitHub](https://github.com/M-TalhaAnsari)
