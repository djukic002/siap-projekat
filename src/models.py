from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def train_model_with_cv(model, X_tr, y_tr, use_weights=False, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_mae_scores = []
    cv_rmse_scores = []
    cv_r2_scores = []

    for train_idx, val_idx in kf.split(X_tr):
        X_f_train, X_f_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_f_train, y_f_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

        # model po foldu
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
            
            # Logaritamsko ublazavanje if
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


def get_final_model_results(model, X_tr, y_tr, X_te, y_te, use_weights=False):
    if use_weights:

        counts = y_tr.round().value_counts().to_dict()

        # inverzna frekvencija
        # w = y_tr.round().apply(lambda val: 1.0 / counts.get(val, 1))
        # w = w / w.mean()
        
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

    results_df = pd.DataFrame({
        "true": y_te,
        "pred": preds
    })

    return results_df

def plot_segmented_analysis(results_df, title="Segmentirana analiza"):
    # Definisanje segmenata
    low = results_df[results_df["true"] < 7]
    mid = results_df[(results_df["true"] >= 7) & (results_df["true"] < 9)]
    high = results_df[results_df["true"] >= 9]

    # Broj uzoraka po segmentu
    counts = [len(low), len(mid), len(high)]
    print("Broj uzoraka po segmentima:")
    print("1-7:", counts[0])
    print("7-9:", counts[1])
    print("9-10:", counts[2])

    # MAE i RMSE po segmentu
    mae_values = [
        mean_absolute_error(low["true"], low["pred"]),
        mean_absolute_error(mid["true"], mid["pred"]),
        mean_absolute_error(high["true"], high["pred"])
    ]
    rmse_values = [
        np.sqrt(mean_squared_error(low["true"], low["pred"])),
        np.sqrt(mean_squared_error(mid["true"], mid["pred"])),
        np.sqrt(mean_squared_error(high["true"], high["pred"]))
    ]
    print("\nRezultati po segmentima:")
    print(f"1-7: MAE = {mae_values[0]:.4f}, RMSE = {rmse_values[0]:.4f}")
    print(f"7-9: MAE = {mae_values[1]:.4f}, RMSE = {rmse_values[1]:.4f}")
    print(f"9-10: MAE = {mae_values[2]:.4f}, RMSE = {rmse_values[2]:.4f}")

    groups = ["1-7", "7-9", "9-10"]
    colors_mae = ['red', 'blue', 'green']
    colors_rmse = ['darkred', 'darkblue', 'darkgreen']

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9,5))
    rects1 = ax.bar(x - width/2, mae_values, width, label='MAE', color=colors_mae, edgecolor='black', alpha=0.8)
    rects2 = ax.bar(x + width/2, rmse_values, width, label='RMSE', color=colors_rmse, edgecolor='black', alpha=0.8)

    ax.set_xlabel("Opseg stvarne vrednosti")
    ax.set_ylabel("Greška")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.show()

def compare_segmented_mae(results1, results2, labels=("Standard", "Weighted"), title="Poređenje MAE po segmentima"):
    segments = [("1-7", 1, 7), ("7-9", 7, 9), ("9-10", 9, 10)]
    mae1, mae2 = [], []
    counts = []

    for _, seg_min, seg_max in segments:
        is_in_segment = (results1["true"] >= seg_min) & (results1["true"] < seg_max)
        if seg_max >= 10:
            is_in_segment = (results1["true"] >= seg_min) & (results1["true"] <= seg_max)

        seg_r1 = results1[is_in_segment]
        seg_r2 = results2[is_in_segment]
        
        mae1.append(mean_absolute_error(seg_r1["true"], seg_r1["pred"]))
        mae2.append(mean_absolute_error(seg_r2["true"], seg_r2["pred"]))
        counts.append(len(seg_r1))

    print("Broj uzoraka i MAE po segmentima:")
    for (label, _, _), m1, m2, c in zip(segments, mae1, mae2, counts):
        print(f"{label}: n={c}, {labels[0]} MAE={m1:.4f}, {labels[1]} MAE={m2:.4f}")

    x = np.arange(len(segments))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, mae1, width, label=labels[0], color='skyblue', edgecolor='black', alpha=0.8)
    ax.bar(x + width/2, mae2, width, label=labels[1], color='salmon', edgecolor='black', alpha=0.8)

    ax.set_xlabel("Segment stvarnih vrednosti (Addiction Level)")
    ax.set_ylabel("MAE")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in segments])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

