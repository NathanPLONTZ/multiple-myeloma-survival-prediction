# Multiple Myeloma Survival Prediction (Machine Learning@NOVA 25)

Predicting the survival time of multiple myeloma patients from clinical features, with missing values and censored observations. This is my work for the Kaggle competition [Machine Learning@NOVA 25: Multiple Myeloma Survival](https://www.kaggle.com/competitions/machine-learning-nova-25-multiple-myeloma-survival), organised as four notebooks (baselines, non-linear models, missing data and censoring, semi-supervised learning).

## Problem

The data is a synthetic dataset simulating clinical records of multiple myeloma patients. The goal is to predict `SurvivalTime` for the patients of the test set.

| Column | Description |
|---|---|
| `Age` | Age of the patient (integer) |
| `Gender` | Biological sex (binary) |
| `Stage` | Extent of the cancer, from 1 (less severe) to 4 (very serious) |
| `GeneticRisk` | Combined genetic cancer risk, real number in [0, 1] |
| `TreatmentType` | Aggressive or milder treatment (binary) |
| `ComorbidityIndex` | Number/severity of additional diseases (integer, 0 = none) |
| `TreatmentResponse` | Effectiveness of the treatment (binary, 0 = poor response) |
| `SurvivalTime` | Target: time from the start of the study/treatment to the event or last follow-up |
| `Censored` | 1 if the survival time is censored (the event was not observed), 0 if the patient died during the study |

The training set has 400 rows and the test set 100 rows. Both contain missing feature values, and part of the training set has no `SurvivalTime` (unlabeled rows), which is what makes the semi-supervised setting relevant.

### Metric

Submissions are scored with the censored mean squared error (cMSE). For a censored patient, only under-predicting the observed time is penalised:

```
cMSE = 1/N * sum_n [ (1 - c_n) * (y_n - y_hat_n)^2 + c_n * max(0, y_n - y_hat_n)^2 ]
```

It is implemented as `error_metric(y, y_hat, c)` in [src/utils.py](src/utils.py). The predicted value must be the second argument.

## Repository structure

```
.
├── data/                 # Place the Kaggle CSV files here (not included, see data/README.md)
├── notebooks/
│   ├── task1.ipynb       # Data exploration and linear regression baseline
│   ├── task2.ipynb       # Non-linear models: polynomial regression and k-NN
│   ├── task3.ipynb       # Missing-value imputation and gradient-boosting / tree models
│   └── task4.ipynb       # Semi-supervised learning (labeled + unlabeled data)
├── src/
│   └── utils.py          # Shared helpers: metric, plots, model selection, boosting CV
├── requirements.txt
└── README.md
```

## Approach

- **Task 1 - Exploration and baseline.** Distribution of each feature, missing-value analysis (`missingno`), relationship between features and survival time. The baseline is a standardised linear regression trained on the uncensored, fully observed rows (`Age`, `Gender`, `Stage`, `TreatmentType`), evaluated with a train/test split and 5-fold cross-validation.
- **Task 2 - Non-linear models.** Polynomial regression (degree selected by cross-validation, degrees 1 to 8) and k-NN regression (k and weighting selected by cross-validation). The stability of the selected hyperparameters is checked over 30 random splits, and the models are compared with the baseline over 5 folds. The notebook also writes the Kaggle submission files.
- **Task 3 - Missing data and censoring.**
  - 3.1: eight imputation strategies (zero, mean, median, most frequent, k-NN uniform, k-NN distance-weighted, two iterative imputers), each with and without missing-value indicator columns, compared with the linear baseline over 10 repeated 5-fold CV runs.
  - 3.2: models that can use all labeled rows, including censored ones: HistGradientBoosting, CatBoost, CatBoost with a survival AFT loss (censored rows are given an open upper bound), XGBoost, LightGBM, plus a decision tree and a random forest. They are evaluated without imputation and with k-NN or iterative imputation.
- **Task 4 - Semi-supervised learning.** The imputer and an Isomap embedding are fitted on labeled and unlabeled rows together, then a linear regression is trained on the labeled rows. A final CatBoost AFT model uses the k-NN imputation of task 3.

## Results

All figures below are copied from the outputs stored in the notebooks. The notebooks use random splits without a fixed seed (except task 4), so numbers change slightly from one run to the next. Tasks 1 and 2 train and score on uncensored rows only (`c = 0`, i.e. a plain MSE), while tasks 3.2 and 4 use the full cMSE on all labeled rows, so the two groups of numbers are not directly comparable.

**Task 1 - linear baseline (uncensored rows, MSE).** Train/test split: 4.54. 5-fold CV: 4.34 and 4.37 in two runs.

**Task 2 - non-linear models (5-fold CV, mean over folds).**

| Model | Mean | Std |
|---|---|---|
| Baseline (linear) | 4.354 | 1.012 |
| Polynomial (degree 1) | 4.410 | 1.428 |
| k-NN (k=16, uniform weights) | 4.325 | 1.144 |

Degree 1 was selected in 27 of 30 repetitions (degree 2 in the other 3). Higher degrees overfit badly (CV error of 733 for degree 5 and millions for degrees 6 to 8). For k-NN, the selected k varied from 8 to 16 across repetitions, almost always with uniform weights. The three models are statistically close given the fold-to-fold variance.

**Task 3.1 - imputation with a linear model (cMSE, mean over 10 runs).** The best and worst methods are shown below; the full ranking is in the notebook.

| Method | Mean | Std |
|---|---|---|
| k-NN (distance), without indicators | 2.523 | 0.057 |
| k-NN (uniform), without indicators | 2.529 | 0.027 |
| k-NN (uniform), with indicators | 2.562 | 0.048 |
| ... | ... | ... |
| Iterative v2, without indicators | 2.689 | 0.024 |
| Iterative v2, with indicators | 2.720 | 0.075 |

The spread between methods (2.52 to 2.72) is small compared with the standard deviations, so the choice of imputer matters little for a linear model.

**Task 3.2 - boosting and tree models (5-fold CV, cMSE, mean over folds).**

| Model | No imputation | k-NN imputation |
|---|---|---|
| HistGradientBoosting | 3.360 | 3.371 |
| CatBoost | 3.513 | 3.546 |
| CatBoost AFT | **2.925** | **3.043** |
| Decision tree | - | 5.097 |
| Random forest | - | 3.266 |
| XGBoost | 3.857 | 3.784 |
| LightGBM | 3.099 | 3.178 |

CatBoost with the survival (AFT) loss is the best model in both settings. The notebook also contains a run with iterative imputation, not summarised here.

**Task 4 - semi-supervised learning (cMSE on the labeled data).** Labeled rows are split into 192 for training and 48 for testing; 160 rows are unlabeled.

| Pipeline | CV error | Test error |
|---|---|---|
| k-NN imputation + scaling + linear regression | 2.704 | 2.437 |
| k-NN imputation + scaling + Isomap (20 components, fitted on labeled + unlabeled) + linear regression | 3.080 | 2.614 |
| k-NN imputation + CatBoost AFT | - | 3.274 |

Among the tested numbers of Isomap components (2, 5, 10, 15, 20), the training error decreased steadily and 20 was kept.

### Kaggle result

Final private leaderboard (about 47% of the test data): **17th out of 41 teams, with a score (cMSE) of 2.50108**, against 2.34358 for the first place. The submitted model was a linear regression with k-NN imputation of the missing values (the approach of tasks 3.1 and 4), which was also the best-performing family in the local cross-validation.

### Limitations

- The imputers in task 3 are fitted on the whole training set before cross-validation, which leaks a little information from the validation folds into the imputation.
- In task 4, the final CatBoost AFT model uses the test split as its `eval_set`, so the model is truncated at iteration 24 using the test data (the best test score of 1.07 at that iteration is not a fair estimate; the reported test error of 3.27 is). Its training error is 1.94, which indicates overfitting.
- The cell outputs saved in the notebooks come from my original runs. After the refactoring, I checked that the code still produces the same printed results and plots when the random seed is fixed.

## Getting started

1. Download the data from Kaggle and place the CSV files in `data/` (see [data/README.md](data/README.md)).
2. Install the dependencies (developed with Python 3.13):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Start Jupyter and run the notebooks from the `notebooks/` folder, in order (`task1` to `task4`):

   ```bash
   cd notebooks
   jupyter notebook
   ```

The notebooks read the data from `../data/` and import `utils` from `../src/`, so they must be started from `notebooks/`. Submission files (`*-submission-*.csv`) and CatBoost training logs (`catboost_info/`) are written to the current folder and are ignored by git.

## License and data

The code is shared for educational purposes. The competition data is provided by the Kaggle competition under the CC BY-NC-SA 4.0 license and is not redistributed here.
