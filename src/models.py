from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def validate_model_with_cv(model, X_tr, y_tr, use_weights=False, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_mae_scores = []
    cv_rmse_scores = []
    cv_r2_scores = []

    for train_idx, val_idx in kf.split(X_tr):
        X_f_train, X_f_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_f_train, y_f_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

        model_fold = model.__class__(**model.get_params())

        if use_weights:
            # linearna funkcija
            # w = 1 + (10 - y_f_train)

            # kvadratna funkcija
            # w = (10 - y_tr) ** 2

            # inverzna frekvencija
            counts = y_f_train.round().value_counts().to_dict()
            # w = y_f_train.round().apply(lambda val: 1.0 / counts.get(val, 1))
            # w = w / w.mean()

            # koren if
            # w = y_f_train.round().apply(lambda val: np.sqrt(1.0 / counts.get(val, 1)))
            # w = w / w.mean()

            # Logaritamsko ublažavanje if
            w = y_f_train.round().apply(lambda val: np.log1p(len(y_f_train) / counts.get(val, 1)))
            w = w / w.mean()

            model_fold.fit(X_f_train, y_f_train, sample_weight=w)
        else:
            model_fold.fit(X_f_train, y_f_train)

        preds_val = model_fold.predict(X_f_val)

        mae_fold = mean_absolute_error(y_f_val, preds_val)
        rmse_fold = np.sqrt(mean_squared_error(y_f_val, preds_val))
        r2_fold = r2_score(y_f_val, preds_val)

        cv_mae_scores.append(mae_fold)
        cv_rmse_scores.append(rmse_fold)
        cv_r2_scores.append(r2_fold)

    avg_mae = np.mean(cv_mae_scores)
    avg_rmse = np.mean(cv_rmse_scores)
    avg_r2 = np.mean(cv_r2_scores)

    print(f"\n===== {model.__class__.__name__} (K-Fold {'Weighted' if use_weights else 'Standard'}) =====")
    print(f"CV Prosečan MAE (Trening): {avg_mae:.4f}")
    print(f"CV Prosečan RMSE (Trening): {avg_rmse:.4f}")
    print(f"CV Prosečan R2 Score (Trening): {avg_r2:.4f}")

    return avg_mae, avg_rmse, avg_r2


def get_final_model_results(model, X_tr, y_tr, X_te, y_te, use_weights=False):
    if use_weights:

        counts = y_tr.round().value_counts().to_dict()

        # Logaritamsko ublažavanje if
        w = y_tr.round().apply(lambda val: np.log1p(len(y_tr) / counts.get(val, 1)))
        w = w / w.mean()

        model.fit(X_tr, y_tr, sample_weight=w)
    else:
        model.fit(X_tr, y_tr)

    preds = model.predict(X_te)

    mae = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2 = r2_score(y_te, preds)

    print(f"\n===== {model.__class__.__name__} (Finalni {'Weighted' if use_weights else 'Standard'}) =====")
    print(f"Finalni MAE (Test): {mae:.4f}")
    print(f"Finalni RMSE (Test): {rmse:.4f}")
    print(f"Finalni R2 Score (Test): {r2:.4f}")

    # DataFrame za segmentiranu analizu
    results_df = pd.DataFrame({
        "true": y_te,
        "pred": preds
    })

    return results_df