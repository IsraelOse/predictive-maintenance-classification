# Predictive Maintenance Classification

Machine-failure prediction on the UCI **AI4I 2020 Predictive Maintenance** dataset (10,000 rows, ~3.4% failure rate), delivered as a **modular, production-style data pipeline** with a companion analysis notebook. Fixed random seed = **93** throughout.

## Problem
Predict machine failure from sensor and process readings (air/process temperature, rotational speed, torque, tool wear). The target is highly imbalanced, which is the central modelling challenge.

## Repository architecture
The pipeline follows a decoupled, block-based layout. Each block is a pure step that accepts inputs and returns outputs, with no hidden coupling between loaders, transformers, and exporters.

```
.
├── data_loaders/        # Fetch/cache raw data (UCI archive)
│   └── ai4i_loader.py
├── transformers/        # Pure cleaning, feature engineering, split, modelling
│   ├── clean.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── split.py
│   └── model_training.py
├── data_exporters/      # Write processed data and metrics
│   └── local_exporter.py
├── utils/               # Shared config + structured logging
│   ├── config.py
│   └── logging_config.py
├── pipelines/
│   └── predictive_maintenance/
│       ├── run.py           # Orchestration: loader -> transformers -> exporter
│       ├── metadata.yaml    # Execution order, variables, schedule
│       └── requirements.txt
├── notebooks/           # Full analysis (EDA, tuning, thresholds, robustness)
├── report/              # Written report
├── data/raw/            # Cached raw dataset (offline, reproducible runs)
└── outputs/             # Generated tables and figures
```

**Coding standards:** every block uses Python type hints and the standard `logging` module (no `print`), and the three I/O block types are fully decoupled.

## Run it
```bash
pip install -r requirements.txt
python -m pipelines.predictive_maintenance.run
```
Runs in a few seconds and writes the model comparison to `outputs/tables/`.

## Approach
- **Leakage control:** the five post-hoc failure-mode flags (TWF/HDF/PWF/OSF/RNF) are dropped — four of them OR together to reproduce the target, so keeping any would hand over the answer.
- **Physics-based features:** `power_W` (torque × angular velocity), `temp_diff_K` (process − air), `wear_torque` (tool wear × torque).
- **Shared preprocessing factory:** one `ColumnTransformer` for all models; standardisation only for logistic regression; transformers fitted inside CV folds, never on full data.
- **Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost; selected on PR-AUC (appropriate for imbalance).

## Results (held-out test set, seed 93)
| Model | ROC-AUC | PR-AUC | Precision | Recall |
|-------|:------:|:-----:|:--------:|:-----:|
| Logistic Regression | 0.909 | 0.547 | 0.833 | 0.221 |
| Decision Tree | 0.971 | 0.885 | 1.000 | 0.779 |
| Random Forest | 0.968 | 0.889 | 0.964 | 0.779 |
| XGBoost | 0.975 | 0.864 | 0.883 | 0.779 |

Random Forest and XGBoost give the strongest ranking performance (best PR-AUC / ROC-AUC) and offer a tunable operating point; the Decision Tree is strongest at a fixed 0.5 cut-off but offers few distinct operating points. The full model selection, threshold analysis, and robustness checks across ten seeds are in the notebook.

## Tech
Python · pandas · scikit-learn · XGBoost · imbalanced-learn

*Project from the Master of Data Analytics program, University of Niagara Falls Canada.*
