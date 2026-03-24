from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import shap
from sklearn.base import clone

def train_model_with_cv(model, X_tr, y_tr, use_weights=False, use_smote=False, use_scaling=False, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_mae_scores, cv_rmse_scores, cv_r2_scores = [], [], []

    modes = []
    if use_scaling: modes.append("Scaled")
    if use_smote: modes.append("SMOTE")
    if use_weights: modes.append("Weighted")
    mode_desc = " + ".join(modes) if modes else "Standard"

    for train_idx, val_idx in kf.split(X_tr):
        X_f_train, X_f_val = X_tr.iloc[train_idx].copy(), X_tr.iloc[val_idx].copy()
        y_f_train, y_f_val = y_tr.iloc[train_idx].reset_index(drop=True), y_tr.iloc[val_idx].reset_index(drop=True)

        if use_scaling:
            scaler = StandardScaler()
            X_f_train = pd.DataFrame(scaler.fit_transform(X_f_train), columns=X_tr.columns)
            X_f_val = pd.DataFrame(scaler.transform(X_f_val), columns=X_tr.columns)
        else:
            X_f_train = X_f_train.reset_index(drop=True)
            X_f_val = X_f_val.reset_index(drop=True)

        if use_smote:
            try:
                temp_df = pd.concat([X_f_train, y_f_train], axis=1)
                y_rounded = y_f_train.round().astype(int)
                
                counts = y_rounded.value_counts()
                if counts.min() > 1:
                    k_neigh = min(5, counts.min() - 1)
                    sm = SMOTE(random_state=42, k_neighbors=k_neigh)
                    temp_resampled, _ = sm.fit_resample(temp_df, y_rounded)
                    
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
    print(f"\n===== {model.__class__.__name__} (K-Fold {mode_desc}) =====")
    print(f"CV MAE: {avg_mae:.4f} | RMSE: {np.mean(cv_rmse_scores):.4f} | R2: {np.mean(cv_r2_scores):.4f}")

    return avg_mae


def train_final_model(model, X_tr, y_tr, use_weights=False, use_smote=False, use_scaling=False):
    model = clone(model)

    X_train = X_tr.copy().reset_index(drop=True)
    y_train = y_tr.copy().reset_index(drop=True)
    scaler = None

    if use_scaling:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_tr.columns)

    if use_smote:
        temp_df = pd.concat([X_train, y_train], axis=1)
        y_rounded = y_train.round().astype(int)
        counts = y_rounded.value_counts()
        if counts.min() > 1:
            k_neigh = min(5, counts.min() - 1)
            sm = SMOTE(random_state=42, k_neighbors=k_neigh)
            temp_resampled, _ = sm.fit_resample(temp_df, y_rounded)
            X_train = temp_resampled.drop(columns=[y_train.name])
            y_train = temp_resampled[y_train.name]

    if use_weights:
        counts = y_train.round().value_counts().to_dict()
        w = y_train.round().apply(lambda val: np.log1p(len(y_train) / counts.get(val, 1)))
        w = w / w.mean()
        model.fit(X_train, y_train, sample_weight=w)
    else:
        model.fit(X_train, y_train)

    return model, scaler

def evaluate_model(model, X_te, y_te, scaler=None):
    X_test = X_te.copy().reset_index(drop=True)

    if scaler is not None:
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_te.columns)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2 = r2_score(y_te, preds)
    
    n, p = X_test.shape
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | Adj R2: {adj_r2:.4f}")
    return pd.DataFrame({"true": y_te, "pred": preds})

def run_shap(model, X_tr, scaler=None, model_type='tree'):
    """
    model_type: 'tree' za XGBoost, 'kernel' za MLP
    """
    X_sample = X_tr.copy().reset_index(drop=True)
    if scaler:
        X_sample = pd.DataFrame(scaler.transform(X_sample), columns=X_tr.columns)

    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        X_summary = shap.kmeans(X_sample, 10)
        explainer = shap.KernelExplainer(model.predict, X_summary)
        shap_values = explainer.shap_values(X_sample.sample(50, random_state=42))
        X_sample = X_sample.sample(50, random_state=42)

    shap.summary_plot(shap_values, X_sample)


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


# modeli sa transformacijama ciljnog obelezja

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
            y_f_train_trans = np.power(y_f_train, 0.75)
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

    mae = mean_absolute_error(y_te, preds_final)
    r2 = r2_score(y_te, preds_final)

    print(f"\n===== {model.__class__.__name__} (Finalni {method.upper()} Transformed) =====")
    print(f"Finalni MAE (Test): {mae:.4f}")
    print(f"Finalni R2 Score (Test): {r2:.4f}")

    return pd.DataFrame({"true": y_te, "pred": preds_final})