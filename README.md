# Pipeline Accidents

Exploratory analysis, statistical testing, feature engineering, model
selection, and clustering on the PHMSA hazardous liquid pipeline accident
dataset — 2,795 reported US accidents from 2010 to 2017.

**Live report:** https://nik8x.github.io/pipeline_accidents/

## Data

`pipeline-accidents.csv` — one row per reported accident, 48 columns covering
operator/location, cause category, release volumes, injuries/fatalities, and
cost breakdowns. Sourced from PHMSA hazardous liquid pipeline accident
reports.

## Notebooks

| Notebook | What it covers |
|---|---|
| `01_exploratory_data_analysis.ipynb` | Missing data, distributions, time trends, correlation between severity columns. |
| `02_statistical_testing.ipynb` | Correlation, Welch's t-test, ANOVA + Tukey HSD, chi-square, and a correlation-vs-causation discussion. |
| `03_feature_engineering_selection.ipynb` | New features (time, shutdown duration, cost/volume ratios, region), narrowed down with mutual information, chi-square, and Random Forest importance. Saves `pipeline_accidents_features.csv`. |
| `04_model_training_evaluation.ipynb` | Train/validation/test split, Logistic Regression vs Random Forest vs Gradient Boosting to predict `Cause Category`, hyperparameter tuning, held-out test evaluation. |
| `05_clustering.ipynb` | KMeans (elbow + silhouette) and DBSCAN clustering of accidents by severity profile. |

Run them in order (01 → 05) — `03` writes `pipeline_accidents_features.csv`,
which `04` and `05` both read.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Key findings

- **Cost and spill volume don't move together year to year.** 2010 had the
  highest total reported cost despite a below-average net loss; 2013 had the
  largest total spill volume with fairly ordinary costs. Yearly totals are
  driven by a handful of outlier incidents, not a steady trend.
- **Underground pipe costs significantly more to fix.** A Welch's t-test on
  log-cost between underground and aboveground accidents is significant at
  p < 0.001 — consistent with the expense of excavation and soil/groundwater
  remediation.
- **Cause category is associated with both cost and pipeline type.** ANOVA
  and a chi-square test of independence are both significant (p ≈ 0), though
  the r ≈ 0.06 raw correlation between spill size and cost is a reminder that
  significance doesn't imply a strong or causal relationship.
- **Predicting accident cause is genuinely hard.** A tuned Random Forest
  reaches macro-F1 ≈ 0.31 against a ≈ 0.10 majority-class baseline — a real
  improvement, but the available features describe accident *consequences*
  (cost, volume, location), not root-cause drivers like pipe age or
  inspection history.
- **Clustering finds a high-risk segment without ever seeing an injury
  label.** KMeans on severity features alone isolates an 8-incident cluster
  with a 75% injury / 100% fatality rate. DBSCAN independently flags
  overlapping incidents as density "noise," with injury and fatality rates
  about 29x the dataset average among the flagged points.

Full detail, plots, and the actual test statistics are in the notebooks and
the [live report](https://nik8x.github.io/pipeline_accidents/).
