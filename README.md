# Preconception Spontaneous Abortion Risk Prediction with Uncertainty Quantification

A machine learning pipeline for predicting spontaneous abortion (SA) risk from preconception examination data, featuring a 300-member XGBoost bootstrap ensemble with Optuna hyperparameter optimization, comprehensive uncertainty quantification, and SHAP-based model interpretability.

---

## Overview

This repository implements a complete ML workflow applied to a large-scale preconception cohort (N ≈ 402,226 records, 134 variables). The pipeline covers data imputation, temporal validation splitting, Boruta feature selection, baseline model benchmarking, bootstrap ensemble training, and TreeSHAP interpretability analysis.

**Outcome variable:** Binary classification — Spontaneous Abortion (SpontAbortion)

**Validation strategy:** Temporal holdout (2014–2018 development / 2019 external validation) + stratified internal validation

---

## Pipeline

```
01_data_preprocessing   →   02_data_splitting   →   03_feature_selection
                                                             ↓
07_shap_analysis        ←   06_model_results    ←   05_bootstrap_ensemble
                                     ↑
                             04_model_training (baseline comparison)
```

| Notebook | Description |
|---|---|
| `01_data_preprocessing.ipynb` | Random Forest imputation of missing values across 137 variables |
| `02_data_splitting.ipynb` | Temporal holdout + stratified 9:1 train/internal-val split |
| `03_feature_selection.ipynb` | Boruta feature selection (500 trees, 100 iterations) |
| `04_model_training.ipynb` | Baseline comparison: LR, RF, XGBoost, LightGBM, MLP |
| `05_bootstrap_ensemble.ipynb` | 300-member XGBoost ensemble with per-model Optuna HPO |
| `06_model_results.ipynb` | Comprehensive evaluation: AUC, calibration, DCA, risk stratification |
| `07_shap_analysis.ipynb` | TreeSHAP feature importance, interaction heatmaps, dependence plots |

---

## Dataset

| Split | Years | Records | SA Rate |
|---|---|---|---|
| Training | 2014–2018 | 360,363 | 3.26% (11,753 events) |
| Internal Validation | 2014–2018 | 40,041 | 3.26% (1,306 events) |
| Temporal Validation | 2019 | 1,822 | 14.38% (262 events) |

> The raw dataset is not included in this repository. Notebooks expect input files under `./Data/`.

---

## Methods

### 1. Data Preprocessing (`01`)
- Block-wise imputation to handle 137-variable missing data efficiently
- Continuous variables: `RandomForestRegressor`
- Categorical variables: `RandomForestClassifier`
- Two modes: (A) single-pass imputation; (B) split-first pipeline to prevent data leakage
- Outputs: pre/post-imputation statistical reports (SMD, missing rates) as Excel workbooks

### 2. Data Splitting (`02`)
- Temporal validation: 2019 held out as external test set
- Development set (2014–2018) stratified by outcome with 90/10 train-val ratio
- Preserves SA prevalence across all three splits

### 3. Feature Selection (`03`)
- **Boruta** algorithm (Random Forest base learner, 500 trees, α = 0.05)
- Trained exclusively on the training set to prevent leakage
- Post-selection: VIF multicollinearity analysis, Pearson/Spearman correlation matrices
- Selected features applied uniformly across all three splits

### 4. Baseline Models (`04`)
Five classifiers benchmarked:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Multi-Layer Perceptron (MLP)

Evaluation: AUC-ROC, AUC-PR (bootstrap 95% CI, N=100), calibration curves, Brier score, Decision Curve Analysis (DCA), Sensitivity/Specificity/PPV/NPV.

### 5. Bootstrap Ensemble (`05`)
- **300 XGBoost models**, each trained on a balanced bootstrap sample (1:1 class resampling)
- Per-model **Optuna** hyperparameter optimization (TPESampler, 50 trials/model)
- GPU (CUDA) acceleration support via automatic device detection
- Models serialized to `./bootstrap_models/model_bs_*.pkl` (included in this repository)

### 6. Model Evaluation (`06`)
- Ensemble predictions via mean/median/voting/weighted aggregation
- Bootstrap 95% CI with **1,000 iterations** for all metrics
- Risk stratification with 4-zone uncertainty framework
- Subgroup analysis by age and residence
- Feature importance: Gain, Weight, mean |SHAP| with rank correlation
- Publication-quality figures exported as 300 DPI TIFF

### 7. SHAP Analysis (`07`)
- **TreeSHAP** computed per model across all 300 ensemble members
- Multi-method importance fusion:
  - Gain (35% weight) + Weight (15% weight) + mean |SHAP| (50% weight)
- Per-split SHAP comparison to assess generalization
- Visualizations: beeswarm plots, interaction heatmaps, dependence plots, stability boxplots
- Bootstrap 95% CI for mean |SHAP| via subsampling (n=3,000)

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna shap boruta \
            matplotlib seaborn statsmodels openpyxl tqdm
```

| Package | Purpose |
|---|---|
| `scikit-learn` | Imputation, baseline models, metrics |
| `xgboost` | Ensemble base learner |
| `lightgbm` | Baseline model |
| `optuna` | Hyperparameter optimization |
| `shap` | TreeSHAP interpretability |
| `boruta` | Feature selection |
| `statsmodels` | VIF analysis |
| `openpyxl` | Excel report generation |
| `tqdm` | Progress tracking |

---

## Directory Structure

```
.
├── Data/
│   ├── dataset1008.csv                        # Raw input (not included)
│   ├── data_training_imputed.csv
│   ├── data_internal_validation_imputed.csv
│   ├── data_temporal_validation_imputed.csv
│   ├── data_training_selected.csv
│   ├── data_internal_validation_selected.csv
│   └── data_temporal_validation_selected.csv
├── bootstrap_models/
│   ├── model_bs_0.pkl
│   ├── model_bs_1.pkl
│   └── ...  (300 models, ~1.8 GB total, ~7.3 MB each)
├── figures/                                   # Publication-quality outputs (300 DPI TIFF)
├── outputs_fi/                                # SHAP outputs and feature importance CSVs
├── 01_data_preprocessing.ipynb
├── 02_data_splitting.ipynb
├── 03_feature_selection.ipynb
├── 04_model_training.ipynb
├── 05_bootstrap_ensemble.ipynb
├── 06_model_results.ipynb
└── 07_shap_analysis.ipynb
```

---

## Usage

### Option A: Use pre-trained models (recommended)

The 300 pre-trained ensemble models are included in `bootstrap_models/`. Skip directly to evaluation and interpretability:

```bash
jupyter notebook 06_model_results.ipynb
jupyter notebook 07_shap_analysis.ipynb
```

### Option B: Full pipeline from scratch

Run notebooks sequentially (requires the raw dataset under `./Data/`):

```bash
jupyter notebook 01_data_preprocessing.ipynb
jupyter notebook 02_data_splitting.ipynb
jupyter notebook 03_feature_selection.ipynb
jupyter notebook 04_model_training.ipynb
jupyter notebook 05_bootstrap_ensemble.ipynb   # Computationally intensive
jupyter notebook 06_model_results.ipynb
jupyter notebook 07_shap_analysis.ipynb
```

> **Note:** `05_bootstrap_ensemble.ipynb` trains 300 XGBoost models with 50 Optuna trials each. GPU (CUDA) acceleration is detected automatically. This step may take several hours on CPU.

---

## Key Results

- **Ensemble size:** 300 XGBoost models
- **Feature count:** ~36 confirmed features (post-Boruta from 134 original variables)
- **Confidence intervals:** Bootstrap 95% CI with 1,000 iterations
- **Evaluation datasets:** Training, Internal Validation (stratified holdout), Temporal Validation (2019)

---

## Citation

If you use this code in your research, please cite accordingly.

---

## License

This project is released for open research use. See [LICENSE](LICENSE) for details.
