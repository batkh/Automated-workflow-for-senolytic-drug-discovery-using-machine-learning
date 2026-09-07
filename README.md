# Senolytic Compound Prediction Pipeline

**Automated workflow for senolytic drug discovery using machine learning**

An end-to-end machine learning benchmark pipeline for identifying and predicting senolytic activity in small molecules, built on **RDKit** and **Mordred** molecular descriptors. The project systematically evaluates multiple feature selection strategies and hyperparameter tuning methods across **12 distinct experimental configurations**.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Overview](#overview)
- [Machine Learning & Software Engineering Best Practices](#machine-learning--software-engineering-best-practices)
- [Dataset](#dataset)
- [Results & Key Findings](#results--key-findings)
- [Repository Structure](#repository-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [Usage](#usage)
- [Generated Outputs & Metrics](#generated-outputs--metrics)
- [Experiment Matrix](#experiment-matrix)
- [Reproducibility](#reproducibility)
- [Limitations & Future Work](#limitations--future-work)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Problem Statement

**Setup.** Let `D = {(x_i, y_i)} for i = 1..N` be a dataset of N compounds, where `x_i ∈ ℝ^d` is a molecular feature vector (RDKit: d ≈ 200; Mordred: d ≈ 1,800) and `y_i ∈ {0, 1}` indicates senolytic activity (1 = senolytic, 0 = non-senolytic). The goal is to learn a classifier `f_θ : ℝ^d → {0, 1}`, parameterized by θ, that generalizes to unseen compounds and correctly predicts y from x alone — reducing, on its surface, to standard supervised binary classification: finding θ that minimizes expected risk under a loss function L:


$$\theta^* = \arg\min_\theta \; \mathbb{E}_{(x,y)\sim D}\big[\, \mathcal{L}(f_\theta(x), y) \,\big]$$



In practice, two properties of D make this minimization ill-posed for classical learners.

**Challenge 1 — Class imbalance.** The dataset exhibits an imbalance ratio:


$$\rho = \frac{|\{i : y_i = 0\}|}{|\{i : y_i = 1\}|} \approx \frac{2{,}465}{58} \approx 42{:}1$$


Under 0–1 loss, the trivial constant classifier `f(x) = 0` for all x achieves accuracy `≈ 1 − 1/(ρ+1) ≈ 97.7%` without learning anything. Any f_θ that merely correlates with the majority class will therefore appear to perform well under accuracy, while precision and recall on the positive class — the metrics that actually matter for candidate prioritization — remain low. This makes accuracy an unreliable objective and motivates minority-sensitive metrics (MCC, F1, recall on the positive class) as the true evaluation target.

**Challenge 2 — Class separability in descriptor space.** Even correcting for imbalance, classification is only tractable if the classes are separable in ℝ^d: there must exist some subspace or decision boundary where between-class variance dominates within-class variance. Fisher's criterion formalizes this along a projection direction w:

$$J(w) = \frac{w^{\top} S_B\, w}{w^{\top} S_W\, w}, \qquad S_B = (\mu_1 - \mu_0)(\mu_1 - \mu_0)^{\top}, \quad S_W = \sum_{c \in \{0,1\}} \sum_{i: y_i = c} (x_i - \mu_c)(x_i - \mu_c)^{\top}$$

Exploratory analysis of the descriptor space shows no direction (or clustering) along which J(w) is appreciably large: known senolytics do not occupy a distinct region relative to non-senolytics. This indicates that the feature distributions of senolytics and non-senolytics largely overlap under RDKit/Mordred representations — i.e., the descriptors may not encode the mechanistic signal underlying senolytic activity, independent of which classifier or feature subset is chosen.

**Research question.** Given a family of feature selection operators `φ_k : ℝ^d → ℝ^k` (k ≪ d; Random Forest importance, Mutual Information, Fisher score) and a family of hypothesis classes H (SVM, Random Forest, Naive Bayes, XGBoost, Logistic Regression) tuned via nested cross-validation over hyperparameters λ:

```
λ*     = argmin_λ  E_val  [ L(f_θ(λ)(φ_k(x)), y) ]
θ(λ)   = argmin_θ  E_train[ L(f_θ(φ_k(x)), y) ]
```

this project performs a grid search over the Cartesian product:

```
{RDKit, Mordred} × {RF-importance, MI, Fisher} × {Baseline, Optuna}
```

(12 configurations) to determine whether classification performance is bottlenecked by **modeling choice** (φ_k, H, λ — improvable via better tuning) or by the **descriptor representation itself** (x — requiring richer features, e.g. mechanism-of-action or 3D/pharmacophore descriptors, to make the classes separable at all).

---

## Overview

The pipeline runs in four stages:

1. **Dataset Loading & Sanitization**
   Reads molecular feature matrices, strips non-feature metadata (e.g. `SMILES`, `Name`), handles string conversion errors, and clips extreme values to prevent numeric overflow.

2. **Feature Selection**
   Reduces high-dimensional descriptor spaces to the most informative features using one of three strategies:
   - **Random Forest Feature Importance**
   - **Mutual Information** (`mutual_info_classif`)
   - **Fisher Score Filtering**

3. **Hyperparameter Tuning & Optimization**
   Evaluates model performance using one of two strategies:
   - **Baseline** — XGBoost with dynamic maximum-depth selection
   - **Optuna** — automated search across multiple algorithms (**SVM**, **Random Forest**, **Naive Bayes**, **XGBoost**, **Logistic Regression**) via nested cross-validation

4. **Performance Evaluation & Logging**
   Runs multiple independent trials per experiment against internal validation splits and external test sets, and generates performance reports as **Mean ± Standard Deviation**.

---

## Machine Learning & Software Engineering Best Practices

| Practice | Description |
|---|---|
| **Nested Cross-Validation** | Prevents data leakage during hyperparameter optimization by decoupling trial evaluation from hyperparameter selection. |
| **Robust Feature Alignment** | Ensures the external test set features match the exact subset and order of features selected from the training set. |
| **Strict Numerical Sanitization** | Prevents runtime failures caused by invalid outputs (`NaN`, `Inf`, string errors) from molecular feature generators like Mordred. |
| **Aggregated Trial Evaluation** | Runs *N* independent trials with varying random seeds to report realistic Mean ± Std performance instead of single-run outcomes. |
| **Data Leakage Mitigation** | Scales features with `MinMaxScaler` inside `Pipeline` objects, so test/validation sets are scaled strictly using training-set parameters. |

---

## Dataset

The datasets are **highly imbalanced**, with senolytic-positive compounds representing only ~2% of all samples:

| No. of Compounds | Initial Training | Final Training | Initial Test Set | Final Test Set |
|---|---:|---:|---:|---:|
| Positives | 58 | 58 | 45 | 38 |
| Negatives | 2,465 | 2,451 | 2,307 | 1,205 |
| **Total** | **2,523** | **2,509** | **2,352** | **1,243** |

This imbalance ($\rho \approx 42{:}1$) is central to interpreting the results below: a classifier predicting "negative" for nearly every compound will still score high accuracy, simply because negatives dominate the dataset.

> **Data source:** _[Add: origin of the compound library and senolytic activity labels — e.g. assay type, publication/database reference.]_

---

## Results & Key Findings

Across all 12 experimental configurations, performance was **consistently low and largely similar regardless of descriptor type, feature selection method, or optimization strategy**. No configuration meaningfully outperformed the others.

**Accuracy vs. precision/recall.** Reported accuracy is consistently high, but precision and recall on the positive (senolytic) class remain low. This gap is a direct consequence of the class imbalance shown above: with positives making up only ~2% of the data, a model can achieve high accuracy while still failing to reliably identify true senolytics. Accuracy alone is therefore not a meaningful metric for this task — MCC, F1-score, and recall on the positive class are more informative and are reported alongside it for this reason.

**Chemical space analysis.** Beyond the modeling results, an analysis of how known senolytics are distributed across chemical space (per the Fisher criterion above) showed **no common trend or clustering** among positive compounds — they do not occupy a distinct, separable region relative to negatives.

**Interpretation.** Taken together, these results suggest that without stronger prior information (e.g. mechanism-based descriptors, target-specific features, or curated structural alerts for senescence pathways), classical machine learning on standard molecular descriptors (RDKit/Mordred) is not well suited to detecting senolytic activity from structure alone. The lack of clustering indicates senolytic activity in this dataset is not driven by a shared, learnable structural signature, which limits what descriptor-based classifiers — regardless of feature selection or hyperparameter tuning — can achieve on this problem.

---

## Repository Structure

```text
.
├── Mordred_datasets/
│   ├── Screening_set_Mordred.csv
│   ├── Test_set_Mordred.csv
│   └── training_set_Mordred.csv
├── RDKit_datasets/
│   ├── Test_set_RDKit.csv
│   └── training_set_RDKit.csv
├── output/                  # Generated CSV metrics and tables
├── cv_env/                  # Virtual environment
├── .gitignore
├── main.py                  # CLI runner for Experiments 01–12
├── requirements.txt         # Project dependencies
└── utils.py                 # Feature selection, preprocessing, and tuning logic
```

---

## Prerequisites & Installation

Requires **Python 3.11+**.

```bash
# Clone the repository
git clone <repo-url>
cd <repo-directory>

# Create and activate a virtual environment
python -m venv cv_env
source cv_env/bin/activate      # macOS/Linux
cv_env\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**

| Library | Purpose |
|---|---|
| `rdkit` | Molecular descriptor generation (~200 features) |
| `mordred` | High-dimensional molecular descriptor generation (~1,800 features) |
| `scikit-learn` | Feature selection, preprocessing pipelines, classical models, cross-validation |
| `xgboost` | Gradient-boosted tree baseline and tunable model |
| `optuna` | Automated multi-model hyperparameter search |
| `pandas` / `numpy` | Data loading, sanitization, and numerical operations |

_(See `requirements.txt` for pinned versions.)_

---

## Usage

Run any experiment by passing its Experiment ID (1–12) on the command line:

```bash
python main.py <experiment_id>
```

**Examples:**

```bash
# Experiment 01 — RDKit baseline with Random Forest Importance
python main.py 1

# Experiment 12 — Mordred with Fisher Score & Optuna hyperparameter optimization
python main.py 12
```

> **Note:** Optuna-based experiments (04–06, 10–12) run nested cross-validation across five model families and are significantly more compute-intensive than baseline experiments, particularly on the ~1,800-feature Mordred set. Expect longer runtimes for these configurations.

---

## Generated Outputs & Metrics

Each experiment run produces the following CSV files:

| File | Contents |
|---|---|
| `Method_XX-Table1.csv` | Aggregated performance metrics (Mean ± Std) across all trials on the internal validation split |
| `Method_XX-Table2.csv` | Aggregated performance metrics (Mean ± Std) across all trials on the external test set |
| `Method_XX-Table1_raw.csv` | Trial-by-trial log for the internal validation split (for auditing) |
| `Method_XX-Table2_raw.csv` | Trial-by-trial log for the external test set (for auditing) |

**Every report logs:**

- **Classification performance** — Accuracy, Precision, Recall, F1-Score
- **Correlation & sensitivity** — Matthews Correlation Coefficient (MCC), False Positive Rate (FPR), ROC-AUC
- **Confusion matrix** — True Positives, True Negatives, False Positives, False Negatives
- **Metadata** — best selected model, number of selected features, hyperparameters

---

## Experiment Matrix

The benchmark systematically combines **2 descriptor types** (RDKit vs. Mordred) × **3 feature selection algorithms** (Random Forest, Mutual Information, Fisher Score) × **2 optimization strategies** (Baseline XGBoost vs. Optuna multi-model search) — 12 configurations in total.

### RDKit Descriptor Experiments (01–06)

*Evaluates ~200 chemical descriptors generated by RDKit.*

| # | Feature Selection | Model Strategy | Goal |
|---|---|---|---|
| 01 | Random Forest Importance (`rf_importance`) | Baseline XGBoost, cross-validated tree depth | Establish a lower-dimensional baseline using tree-based feature selection |
| 02 | Mutual Information (`mutual_info_classif`) | Baseline XGBoost, cross-validated tree depth | Test whether non-linear dependency filtering captures better molecular signals than tree importance |
| 03 | Fisher Score (dynamic thresholds) | Baseline XGBoost, cross-validated tree depth | Measure performance across variance-ratio feature selection thresholds |
| 04 | Random Forest Importance (`rf_importance`) | Optuna nested CV (SVM, RF, Naive Bayes, XGB, LR) | Identify the best architecture and hyperparameters for RF-selected RDKit features |
| 05 | Mutual Information (`mutual_info_classif`) | Optuna nested CV (SVM, RF, Naive Bayes, XGB, LR) | Benchmark multi-model hyperparameter search using Mutual Information features |
| 06 | Fisher Score (dynamic thresholds) | Optuna nested CV (SVM, RF, Naive Bayes, XGB, LR) | Determine optimal classifier performance using Fisher Score features |

### Mordred Descriptor Experiments (07–12)

*Evaluates ~1,800 high-dimensional chemical descriptors generated by Mordred.*

| # | Feature Selection | Model Strategy | Goal |
|---|---|---|---|
| 07 | Random Forest Importance (`rf_importance`) | Baseline XGBoost, cross-validated tree depth | Establish a high-dimensional baseline using tree-based feature selection |
| 08 | Mutual Information (`mutual_info_classif`) | Baseline XGBoost, cross-validated tree depth | Evaluate Mutual Information filtering on complex, high-dimensional descriptor spaces |
| 09 | Fisher Score (dynamic thresholds) | Baseline XGBoost, cross-validated tree depth | Test Fisher Score feature subset selection on Mordred features |
| 10 | Random Forest Importance (`rf_importance`) | Optuna nested CV (SVM, RF, Naive Bayes, XGB, LR) | Search for optimal algorithm and hyperparameters on Mordred features |
| 11 | Mutual Information (`mutual_info_classif`) | Optuna nested CV (SVM, RF, Naive Bayes, XGB, LR) | Benchmark hyperparameter tuning on Mutual Information-selected Mordred features |
| 12 | Fisher Score (dynamic thresholds) | Optuna nested CV (SVM, RF, Naive Bayes, XGB, LR) | Maximize predictive performance on Fisher Score-selected Mordred features |

---

## Reproducibility

- Each experiment aggregates results over *N* independent trials with varying random seeds — see `Method_XX-Table1_raw.csv` / `Method_XX-Table2_raw.csv` for per-trial values.
- Feature scaling (`MinMaxScaler`) and feature selection are fit exclusively on training folds and applied to validation/test folds to avoid leakage.
- Nested cross-validation is used for all Optuna-based experiments to separate hyperparameter selection from performance estimation.
- Exact seed values, trial count ($N$), and CV fold count are configured in `main.py` / `utils.py`. _[Add: specific values, e.g. N=10 trials, 5-fold CV, seeds 0–9.]_

---

## Limitations & Future Work

- **Severe class imbalance** (~2% positives, $\rho \approx 42{:}1$) limits what any classical classifier can learn from this dataset, regardless of tuning effort.
- **No clear structural clustering** was observed among known senolytics (low Fisher separability), suggesting that 2D descriptor-based representations (RDKit/Mordred) may not capture the features that actually drive senolytic activity.
- Future directions could include:
  - Incorporating mechanism-of-action or target-based features
  - Exploring 3D/pharmacophore descriptors or learned molecular representations (e.g. graph neural networks)
  - Using imbalance-aware techniques (e.g. SMOTE, class-weighted losses, anomaly-detection framings, one-class classification)
  - Expanding the positive-class dataset as more senolytics are experimentally validated

---


## License

_MIT_

---

