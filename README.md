# Predictive Maintenance Classification

Machine-failure prediction on the UCI **AI4I 2020 Predictive Maintenance** dataset, built as a fully reproducible, script-driven pipeline (fixed random seed = 93).

## Problem
Predict whether a machine will fail from sensor and process readings (air/process temperature, rotational speed, torque, tool wear). The target is heavily imbalanced, which is the central modelling challenge.

## Approach
A staged pipeline, one script per stage (`src/00`–`src/12`):
data audit → EDA → preprocessing → feature engineering → baseline → model tuning → imbalance study → evaluation → threshold analysis → misclassification review → interpretation → robustness checks.

- **Models:** Logistic Regression baseline, Random Forest, XGBoost (with hyperparameter tuning)
- **Imbalance handling:** resampling / class weighting, compared head-to-head
- **Decision quality:** threshold optimisation rather than a default 0.5 cut-off
- **Trust:** dedicated reproducibility, parity, and fabrication-check scripts (`reproducibility_check.py`, `notebook_parity_check.py`, `fabrication_check.py`)

## Repo layout
| Path | What it is |
|------|-----------|
| `src/` | Numbered pipeline stages (run in order) |
| `notebooks/` | Consolidated analysis notebook |
| `report/` | Written report (.docx) |
| `outputs/figures`, `outputs/tables` | Generated results |
| `requirements.txt` | Python dependencies |

## Run it
```bash
pip install -r requirements.txt
# run stages in order, e.g.
python src/01_data_audit.py
```

## Tech
Python · pandas · scikit-learn · XGBoost · matplotlib/seaborn

*Individual take-home assignment, DAMO-510 Predictive Analytics, Master of Data Analytics, University of Niagara Falls Canada.*
