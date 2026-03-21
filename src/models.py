from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import smogn
from imblearn.over_sampling import SMOTE

def train_model_with_cv(model, X_tr, y_tr, use_weights=False, use_smote=False, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_mae_scores = []
    cv_rmse_scores = []
    cv_r2_scores = []

    modes = []
    if use_smote: modes.append("SMOTE")
    if use_weights: modes.append("Weighted")
    mode_desc = " + ".join(modes) if modes else "Standard"

    for train_idx, val_idx in kf.split(X_tr):
        X_f_train, X_f_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_f_train, y_f_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

        if use_smote:
            try:
                # TRIK: Spajamo X i y da bi SMOTE interpolisao i ciljnu promenljivu
                temp_df = pd.concat([X_f_train, y_f_train], axis=1)
                y_rounded = y_f_train.round().astype(int)
                
                counts = y_rounded.value_counts()
                if counts.min() > 1:
                    k_neigh = min(5, counts.min() - 1)
                    sm = SMOTE(random_state=42, k_neighbors=k_neigh)
                    
                    # Resamplujemo temp_df koristeći y_rounded kao klase
                    temp_resampled, _ = sm.fit_resample(temp_df, y_rounded)
                    
                    # Razdvajamo nazad na X i y
                    X_f_train = temp_resampled.drop(columns=[y_f_train.name])
                    y_f_train = temp_resampled[y_f_train.name]
                else:
                    pass 
            except Exception as e:
                print(f"Greška u SMOTE aplikaciji: {e}")

        # model po foldu
        model_fold = model.__class__(**model.get_params())

        if use_weights:
            counts = y_f_train.round().value_counts().to_dict()

            # Dodavanje tezine: log ublazanje if 
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

    print(f"\n===== {model.__class__.__name__} (K-Fold {mode_desc}) =====")
    print(f"CV Prosečan MAE (Trening): {avg_mae:.4f}")
    print(f"CV Prosečan RMSE (Trening): {avg_rmse:.4f}")
    print(f"CV Prosečan R2 Score (Trening): {avg_r2:.4f}")


def get_final_model_results(model, X_tr, y_tr, X_te, y_te, use_weights=False, use_smote=False):
    modes = []
    if use_smote: modes.append("SMOTE")
    if use_weights: modes.append("Weighted")
    mode_desc = " + ".join(modes) if modes else "Standard"

    # Kopiramo podatke da ne bismo menjali originalne varijable van funkcije
    X_train_final = X_tr.copy()
    y_train_final = y_tr.copy()

    if use_smote:
        try:
            temp_df = pd.concat([X_train_final, y_train_final], axis=1)
            y_rounded = y_train_final.round().astype(int)
            
            counts = y_rounded.value_counts()
            if counts.min() > 1:
                k_neigh = min(5, counts.min() - 1)
                sm = SMOTE(random_state=42, k_neighbors=k_neigh)
                
                temp_resampled, _ = sm.fit_resample(temp_df, y_rounded)
                
                X_train_final = temp_resampled.drop(columns=[y_train_final.name])
                y_train_final = temp_resampled[y_train_final.name]
        except Exception as e:
            print(f"SMOTE Greška na finalnom setu: {e}")

    # 3. Treniranje modela (sa težinama ili bez)
    if use_weights:
        counts = y_train_final.round().value_counts().to_dict()
        w = y_train_final.round().apply(lambda val: np.log1p(len(y_train_final) / counts.get(val, 1)))
        w = w / w.mean()
        model.fit(X_train_final, y_train_final, sample_weight=w)
    else:
        model.fit(X_train_final, y_train_final)

    # 4. Predviđanje na TEST setu (test set se nikada ne dira sa SMOTE-om)
    preds = model.predict(X_te)

    mae = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2 = r2_score(y_te, preds)

    print(f"\n===== {model.__class__.__name__} (Finalni {mode_desc}) =====")
    print(f"Finalni MAE (Test):  {mae:.4f}")
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

def plot_compare_rolling_mae(results_dict, window=50):
    """
    Prikazuje pokretni prosek MAE za više modela odjednom.
    
    Argumenti:
    results_dict -- rečnik oblika {"Naziv Modela": results_df}
                    gde results_df mora imati kolone 'true' i 'pred'
    window -- veličina prozora (broj uzoraka) za prosek
    """
    plt.figure(figsize=(12, 6))
    
    for label, df in results_dict.items():
        # 1. Kreiramo privremeni DF sa apsolutnom greškom
        temp = pd.DataFrame({
            'true': df['true'], 
            'error': np.abs(df['true'] - df['pred'])
        })
        
        # 2. Sortiramo po stvarnim vrednostima (X osa)
        temp = temp.sort_values('true')
        
        # 3. Računamo rolling prosek greške
        temp['rolling_mae'] = temp['error'].rolling(window=window, center=True).mean()
        
        # 4. Crtamo liniju za taj model
        plt.plot(temp['true'], temp['rolling_mae'], label=label, linewidth=2)

    # Estetika grafikona
    plt.title(f'Analiza greške: Pokretni MAE (Prozor = {window} uzoraka)', fontsize=14)
    plt.xlabel('Stvarni Addiction Level (Skala 1-10)', fontsize=12)
    plt.ylabel('Srednja Apsolutna Greška (MAE)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Opciono: Postavljamo X osu da uvek bude 1-10 radi preglednosti
    plt.xlim(1, 10)
    
    plt.show()

def plot_error_distribution(results_dict):
    plt.figure(figsize=(12, 6))
    for label, df in results_dict.items():
        sns.kdeplot(df['true'] - df['pred'], label=label, fill=True, alpha=0.2)
    
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Distribucija grešaka (Stvarno - Predviđeno)')
    plt.xlabel('Veličina greške')
    plt.ylabel('Gustina')
    plt.legend()
    plt.show()


# if use_weights:
#             # linearna funkcija
#             # w = 1 + (10 - y_f_train)

#             # kvadratna funkcija
#             # w = (10 - y_tr) ** 2

#             # inverzna frekvencija
#             counts = y_f_train.round().value_counts().to_dict()
#             # w = y_f_train.round().apply(lambda val: 1.0 / counts.get(val, 1))
#             # w = w / w.mean()

#             # koren if
#             # w = y_f_train.round().apply(lambda val: np.sqrt(1.0 / counts.get(val, 1)))
#             # w = w / w.mean()
            
#             # Logaritamsko ublazavanje if
#             w = y_f_train.round().apply(lambda val: np.log1p(len(y_f_train) / counts.get(val, 1)))
#             w = w / w.mean()

#             model_fold.fit(X_f_train, y_f_train, sample_weight=w)
#         else:
#             model_fold.fit(X_f_train, y_f_train)

def train_model_with_transformation_cv(model, X_tr, y_tr, method='sqrt', n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_mae_scores, cv_rmse_scores, cv_r2_scores = [], [], []

    for train_idx, val_idx in kf.split(X_tr):
        X_f_train, X_f_val = X_tr.iloc[train_idx].copy(), X_tr.iloc[val_idx].copy()
        y_f_train, y_f_val = y_tr.iloc[train_idx].copy(), y_tr.iloc[val_idx].copy()

        # Primena izabrane transformacije
        if method == 'sqrt':
            y_f_train_trans = np.sqrt(y_f_train)
        elif method == 'log':
            y_f_train_trans = np.log1p(y_f_train)
        elif method == 'power':
            y_f_train_trans = np.power(y_f_train, 0.75) # Blaža verzija korena
        else:
            y_f_train_trans = y_f_train

        model_fold = model.__class__(**model.get_params())
        model_fold.fit(X_f_train, y_f_train_trans)

        preds_trans = model_fold.predict(X_f_val)

        # Inverzna transformacija
        if method == 'sqrt':
            preds_final = np.square(preds_trans)
        elif method == 'log':
            preds_final = np.expm1(preds_trans)
        elif method == 'power':
            preds_final = np.power(preds_trans, 1/0.75)
        else:
            preds_final = preds_trans
        
        preds_final = np.clip(preds_final, 1, 10)

        cv_mae_scores.append(mean_absolute_error(y_f_val, preds_final))
        cv_rmse_scores.append(np.sqrt(mean_squared_error(y_f_val, preds_final)))
        cv_r2_scores.append(r2_score(y_f_val, preds_final))

    print(f"\n===== {model.__class__.__name__} (K-Fold {method.upper()} Transformed) =====")
    print(f"CV Prosečan MAE: {np.mean(cv_mae_scores):.4f}")
    print(f"CV Prosečan R2 Score: {np.mean(cv_r2_scores):.4f}")


def get_final_model_transformation(model, X_tr, y_tr, X_te, y_te, method='sqrt'):
    X_tr_copy = X_tr.copy()
    y_tr_copy = y_tr.copy()

    # Primena transformacije
    if method == 'sqrt':
        y_tr_trans = np.sqrt(y_tr_copy)
    elif method == 'log':
        y_tr_trans = np.log1p(y_tr_copy)
    elif method == 'power':
        y_tr_trans = np.power(y_tr_copy, 0.75)
    else:
        y_tr_trans = y_tr_copy
    
    model.fit(X_tr_copy, y_tr_trans)
    preds_trans = model.predict(X_te)
    
    # Inverzna transformacija
    if method == 'sqrt':
        preds_final = np.square(preds_trans)
    elif method == 'log':
        preds_final = np.expm1(preds_trans)
    elif method == 'power':
        preds_final = np.power(preds_trans, 1/0.75)
    else:
        preds_final = preds_trans

    preds_final = np.clip(preds_final, 1, 10)

    mae = mean_absolute_error(y_te, preds_final)
    r2 = r2_score(y_te, preds_final)

    print(f"\n===== {model.__class__.__name__} (Finalni {method.upper()} Transformed) =====")
    print(f"Finalni MAE (Test): {mae:.4f}")
    print(f"Finalni R2 Score (Test): {r2:.4f}")

    return pd.DataFrame({"true": y_te, "pred": preds_final})