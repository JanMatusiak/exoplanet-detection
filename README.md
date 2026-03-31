# exoplanet-detection

## 1. Project Summary
This project is an end-to-end binary classification workflow for exoplanet candidate detection using tabular astronomy features from KOI (Kepler Objects of Interest) and K2P (K2 Planets & Candidates).  
The target is `label` where `1` means planet-like and `0` means non-planet-like.

## 2. Goals
- Build a full ML workflow from raw data preparation to deployment-ready artifacts.
- Compare multiple model families with consistent preprocessing.
- Tune decision thresholds for different optimization profiles (`f2`, recall-priority, precision-priority).
- Provide explainability and visual diagnostics for deployed models.

## 3. Data
- Raw files:
  - `data/raw/KOI_full.csv`
  - `data/raw/K2P_full.csv`
- Processed files:
  - `data/processed/KOI_train_set.csv`
  - `data/processed/KOI_test_set.csv`
  - `data/processed/K2P_set.csv`
- Label mapping:
  - `CONFIRMED -> 1`
  - `FALSE POSITIVE -> 0`
  - `REFUTED -> 0`
  - `CANDIDATE` is excluded from supervised training.
- Group-aware split:
  - `group_id` is used in `StratifiedGroupKFold` to reduce host-level leakage risk.
- Full mapped feature descriptions:
  - `FEATURES.md`
- Final model training features:
  - The training pipeline uses `FINAL_FEATURE_COLUMNS` (defined in `src/exoplanet_detector/features/feature_selection.py`), not all mapped columns.

## 4. Pipeline Overview (Notebook Map)
- `notebooks/01_data_preparation.ipynb`
  - Harmonization, renaming, label mapping, KOI train/test split.
- `notebooks/02_feature_selection_and_preprocessing.ipynb`
  - Feature screening, transformations, preprocessing decisions.
- `notebooks/03_train_and_tune_models.ipynb`
  - Randomized hyperparameter search across 5 model families.
- `notebooks/04_evaluate_models.ipynb`
  - Threshold tuning, KOI/K2P evaluation, deployment bundle export.
- `notebooks/05_feature_analysis.ipynb`
  - Permutation importance matrix and SHAP-based single-row explanation.
- `notebooks/06_visualisations.ipynb`
  - Confusion/ROC/PR plot generation and visualization manifest.

## 5. Modeling Approach
- Candidate models:
  - Logistic Regression (`logreg`)
  - SVC with RBF kernel (`svc_rbf`)
  - KNN (`knn`)
  - Random Forest (`rf`)
  - HistGradientBoosting (`hgb`)
- Cross-validation:
  - `StratifiedGroupKFold` on KOI train set.
- Objective during model search:
  - Multi-metric scoring with refit on `f2`.
- Threshold tuning profiles (`TunedThresholdClassifierCV`):
  - `f2`
  - `recall_constrained` (requires minimum precision floor)
  - `precision_constrained` (requires minimum recall floor)

## 6. Current Results (v1)
Deployed profiles from `artifacts/deployment/v1/deploy_manifest.csv`:

| deploy_id | model | profile | threshold | KOI F2 | KOI recall | KOI precision | K2P F2 | K2P recall | K2P precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `deploy_f2` | `knn` | `f2` | 0.3636 | 0.9167 | 0.9581 | 0.7816 | 0.8857 | 0.9086 | 0.8046 |
| `deploy_recall` | `rf` | `recall_constrained` | 0.0290 | 0.8488 | 0.9982 | 0.5310 | 0.9217 | 0.9983 | 0.7052 |
| `deploy_precision` | `hgb` | `precision_constrained` | 0.8854 | 0.5658 | 0.5118 | 0.9791 | 0.2881 | 0.2448 | 0.9861 |

Interpretation:
- `deploy_f2` provides the best overall balance.
- `deploy_recall` maximizes recall with expected precision tradeoff.
- `deploy_precision` maximizes precision with expected recall tradeoff.

## 7. Artifacts & Reproducibility
Artifacts are versioned by `RUN_TAG` (default: `v1`) under `artifacts/`:

- `artifacts/model_search/<run_tag>/`
  - `search_results.joblib`
  - `cv_summary.csv`
- `artifacts/evaluation/<run_tag>/`
  - `tuned_threshold_models.joblib`
  - `threshold_tuning_summary.csv`
  - `comparison_df.csv`
- `artifacts/deployment/<run_tag>/`
  - `deploy_models.joblib`
  - `deploy_manifest.csv`
- `artifacts/visualization/<run_tag>/`
  - per-model/per-dataset confusion/ROC/PR plots
  - `plot_manifest.csv`
- `artifacts/feature_analysis/<run_tag>/`
  - permutation importance outputs and metadata

Important:
- Keep a stable `run_tag` for demo consistency.
- Use a new tag (`v2`, `v3`, ...) when retraining to avoid mixing artifacts.

## 8. How to Run
### Environment setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Execute notebooks in order
1. `01_data_preparation.ipynb`
2. `02_feature_selection_and_preprocessing.ipynb`
3. `03_train_and_tune_models.ipynb`
4. `04_evaluate_models.ipynb`
5. `05_feature_analysis.ipynb`
6. `06_visualisations.ipynb`

### Notes
- Run from project root or `notebooks/`; notebooks include path handling for `src/`.
- If you retrain/evaluate intentionally, set the appropriate force flags or bump `RUN_TAG`.

## 9. Limitations
- Model/profile deployment choices were made after reviewing KOI test and K2P comparison metrics, so reported metrics may be mildly optimistic versus a strict untouched final holdout protocol.
- This repository is designed for portfolio learning and reproducible experimentation, not operational or scientific production use.
