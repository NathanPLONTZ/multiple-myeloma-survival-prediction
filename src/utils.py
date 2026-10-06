"""Shared helpers for the Multiple Myeloma survival notebooks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Metric and data cleaning
# ---------------------------------------------------------------------------

def error_metric(y, y_hat, c):
    """Censored mean squared error (cMSE) used by the competition.

    Censored samples (c=1) are only penalised when the prediction is below the
    observed time.
    """
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(err) / err.shape[0]


def clean_train_data(train):
    """Drop columns with missing values, rows without a label and censored rows."""
    cols_with_missing = [col for col in train.columns if train[col].isna().sum() > 0 and col != "SurvivalTime"]
    train_clean = train.drop(columns=cols_with_missing)
    train_clean = train_clean.dropna(subset=["SurvivalTime"])
    train_clean = train_clean[train_clean["Censored"] == 0]
    return train_clean


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_y_yhat(y, yhat):
    """Scatter plot of true vs predicted values with the y = y_hat line."""
    y = np.array(y)
    yhat = np.array(yhat)

    plt.figure(figsize=(6, 6))
    plt.scatter(y, yhat, alpha=0.6)
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("y vs yhat")

    # Draw y=x line
    x_min, x_max = np.min(y), np.max(y)
    plt.plot([x_min, x_max], [x_min, x_max], color='red')

    plt.axis('square')
    plt.show()


def plot_y_vs_yhat(y_true, y_pred, title="y vs y_hat"):
    """Same as plot_y_yhat, with a custom title, a grid and a dashed diagonal."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)

    # Perfect-prediction diagonal y = y_hat
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Linear baseline
# ---------------------------------------------------------------------------

def baseline_cv_predict(df, keep_missing=True, n_splits=5):
    """Linear-regression baseline with out-of-fold predictions.

    Returns the fitted fold models, the out-of-fold predictions, the cMSE and a
    DataFrame with the predictions. If keep_missing is False, the "*_missing"
    indicator columns are dropped first.
    """
    if not keep_missing:
        df = df.drop(columns=[col for col in df.columns if "_missing" in col])

    # Define features and target
    X = df.drop(columns=["id", "SurvivalTime", "Censored"])
    y = df["SurvivalTime"]
    censored = df["Censored"]
    ids = df["id"]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    kf = KFold(n_splits=n_splits, shuffle=True)
    models = []
    y_pred_total = np.zeros(len(X))

    for train_idx, test_idx in kf.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        model = clone(pipeline)
        model.fit(X_tr, y_tr)
        models.append(model)

        # Predict on the held-out fold
        X_te = X.iloc[test_idx]
        y_pred_total[test_idx] = model.predict(X_te)

    mse = error_metric(y.values, y_pred_total, censored.values)

    df_pred = pd.DataFrame({
        "id": ids,
        "SurvivalTime": y,
        "Censored": censored,
        "y_pred": y_pred_total
    })

    return models, y_pred_total, mse, df_pred


# ---------------------------------------------------------------------------
# Model selection (Task 2)
# ---------------------------------------------------------------------------

def select_polynomial_model(
    X_tr, y_tr,
    degrees=[1, 2, 3, 4, 5],
    n_splits=5,
    shuffle=True
):
    """Pick the polynomial degree with the best cross-validated error.

    Returns (best_model_fitted, best_degree, cv_scores).
    """
    cv_scores = {}
    best_score = np.inf
    best_degree = None
    best_model = None

    for d in degrees:
        # Pipeline: standardisation -> polynomial expansion -> linear regression
        pipe_poly = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("lin", LinearRegression())
        ])

        kfold = KFold(n_splits=n_splits, shuffle=shuffle)

        # Out-of-fold predictions on the training set
        y_tr_oof = cross_val_predict(pipe_poly, X_tr, y_tr, cv=kfold)

        cmse_cv = error_metric(y_tr, y_tr_oof, c=0)
        cv_scores[d] = cmse_cv

        # Keep track of the best model and refit it on the full training set
        if cmse_cv < best_score:
            best_score = cmse_cv
            best_degree = d
            best_model = pipe_poly.fit(X_tr, y_tr)

    return best_model, best_degree, cv_scores


def select_knn_model(
    X_tr, y_tr,
    k_list=[3, 5, 7, 9, 11],
    weights_list=["uniform", "distance"],
    n_splits=5,
    shuffle=True
):
    """Pick the k-NN hyperparameters with the best cross-validated error.

    Returns (best_model_fitted, best_params, cv_scores).
    """
    cv_scores = {}
    best_score = np.inf
    best_params = None
    best_model = None

    for k in k_list:
        for w in weights_list:
            pipe_knn = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(
                    n_neighbors=k,
                    weights=w
                ))
            ])

            kfold = KFold(n_splits=n_splits, shuffle=shuffle)

            # Out-of-fold predictions on the training set
            y_tr_oof = cross_val_predict(pipe_knn, X_tr, y_tr, cv=kfold)

            cmse_cv = error_metric(y_tr, y_tr_oof, c=0)
            cv_scores[(k, w)] = cmse_cv

            if cmse_cv < best_score:
                best_score = cmse_cv
                best_params = {"k": k, "weights": w}
                best_model = pipe_knn.fit(X_tr, y_tr)

    return best_model, best_params, cv_scores


def evaluate_model_cv(model, X, y, n_splits=5, shuffle=True):
    """Return the per-fold cMSE (c=0) of a model under K-fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    fold_errors = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)

        y_hat = m.predict(X_te)
        err = error_metric(y_te.values, y_hat, c=0)
        fold_errors.append(err)

    return fold_errors


# ---------------------------------------------------------------------------
# Gradient boosting and tree models (Task 3.2)
# ---------------------------------------------------------------------------

def run_boosting_cv(X_valid, y_valid, c_valid, include_trees=True):
    """5-fold CV of tree-based regressors on the labelled rows.

    Models: HGB, CatBoost, CatBoost AFT (censoring-aware), XGBoost, LightGBM and,
    if include_trees is True, a decision tree and a random forest.

    Returns (models_dict, y_true_all) where models_dict maps each model name to
    its fold models, per-fold errors (cMSE) and out-of-fold predictions.
    """
    # Preparation for CatBoost AFT: censored rows have an open upper bound
    y_lower = y_valid.values.copy()
    y_upper = y_valid.values.copy()
    censored_mask = (c_valid == 1).values
    y_upper[censored_mask] = np.inf

    kf = KFold(n_splits=5, shuffle=True)

    names = ["HGB", "CatBoost", "CatBoost_AFT"]
    if include_trees:
        names += ["DecisionTree", "RandomForest"]
    names += ["XGBoost", "LightGBM"]
    models_dict = {name: {"models": [], "errors": [], "predictions": []} for name in names}

    def record(name, model, y_val, y_pred, c_val):
        models_dict[name]["models"].append(model)
        models_dict[name]["errors"].append(error_metric(y_val.values, y_pred, c_val.values))
        models_dict[name]["predictions"].append(y_pred)

    y_true_all = []

    for train_idx, val_idx in kf.split(X_valid):
        X_train, X_val = X_valid.iloc[train_idx], X_valid.iloc[val_idx]
        y_train, y_val = y_valid.iloc[train_idx], y_valid.iloc[val_idx]
        c_val = c_valid.iloc[val_idx]

        y_true_all.append(y_val.values)

        # --- HistGradientBoostingRegressor ---
        hgb = HistGradientBoostingRegressor(max_iter=200)
        hgb.fit(X_train, y_train)
        record("HGB", hgb, y_val, hgb.predict(X_val), c_val)

        # --- CatBoostRegressor ---
        cat = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0)
        cat.fit(X_train, y_train)
        record("CatBoost", cat, y_val, cat.predict(X_val), c_val)

        # --- CatBoost AFT (inf -> -1 encodes an open upper bound) ---
        y_train_lower = y_lower[train_idx]
        y_train_upper = y_upper[train_idx].copy()
        y_val_lower = y_lower[val_idx]
        y_val_upper = y_upper[val_idx].copy()
        y_train_upper[y_train_upper == np.inf] = -1
        y_val_upper[y_val_upper == np.inf] = -1

        train_pool = Pool(
            data=X_train,
            label=np.column_stack([y_train_lower, y_train_upper])
        )
        val_pool = Pool(
            data=X_val,
            label=np.column_stack([y_val_lower, y_val_upper])
        )

        cat_aft = CatBoostRegressor(
            loss_function='SurvivalAft:dist=Normal',  # Normal / Logistic / Extreme
            eval_metric='SurvivalAft',
            iterations=500,
            learning_rate=0.1,
            depth=6,
            verbose=0
        )
        cat_aft.fit(train_pool, eval_set=val_pool)
        record("CatBoost_AFT", cat_aft, y_val, cat_aft.predict(val_pool, prediction_type='Exponent'), c_val)

        if include_trees:
            # --- DecisionTreeRegressor ---
            dt = DecisionTreeRegressor()
            dt.fit(X_train, y_train)
            record("DecisionTree", dt, y_val, dt.predict(X_val), c_val)

            # --- RandomForestRegressor ---
            rf = RandomForestRegressor(n_estimators=1000)
            rf.fit(X_train, y_train)
            record("RandomForest", rf, y_val, rf.predict(X_val), c_val)

        # --- XGBoostRegressor ---
        xgb = XGBRegressor(n_estimators=500, learning_rate=0.1)
        xgb.fit(X_train, y_train)
        record("XGBoost", xgb, y_val, xgb.predict(X_val), c_val)

        # --- LightGBMRegressor ---
        lgbm = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, verbose=-1)
        lgbm.fit(X_train, y_train)
        record("LightGBM", lgbm, y_val, lgbm.predict(X_val), c_val)

    return models_dict, y_true_all


def report_boosting_cv(models_dict, y_true_all):
    """Print the cross-validation summary and plot y vs y_hat for every model."""
    print("=" * 60)
    print("RESULTS OVER CROSS-VALIDATION")
    print("=" * 60)
    for name, info in models_dict.items():
        mean_err = np.mean(info['errors'])
        std_err = np.std(info['errors'])
        print(f"{name:15s} | Mean: {mean_err:.6f} | Std: {std_err:.6f}")

    best_model = min(models_dict.items(), key=lambda x: np.mean(x[1]['errors']))
    print(f"\nBest model: {best_model[0]}")
    print(f"   Mean Error: {np.mean(best_model[1]['errors']):.6f}")

    y_true_concat = np.concatenate(y_true_all)

    print("\n" + "=" * 60)
    print("PREDICTION PLOTS (y vs yhat - average of 5 folds)")
    print("=" * 60)

    for name, info in models_dict.items():
        y_pred_concat = np.concatenate(info['predictions'])
        mean_err = np.mean(info['errors'])
        print(f"\n{'=' * 60}")
        print(f"{name} (Error: {mean_err:.6f})")
        print(f"{'=' * 60}")
        plot_y_yhat(y_true_concat, y_pred_concat)
