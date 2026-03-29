import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        print(f"Ukupan broj nedostajućih vrednosti: {total_missing}")
    else:
        print("Nema nedostajućih vrednosti.")

    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        print(f"\nUklanjanje {num_duplicates} duplikata.")
        df.drop_duplicates(inplace=True)
    else:
        print("\nNema duplikata.")

    categorical_cols = ['Gender', 'Phone_Usage_Purpose']

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df_encoded

def merge_datasets(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    target_columns = new_df.columns.tolist()

    old_aligned = old_df[[col for col in target_columns if col in old_df.columns]]

    for col in target_columns:
        if col not in old_aligned.columns:
            old_aligned[col] = None

    old_aligned = old_aligned[target_columns]

    merged_df = pd.concat([old_aligned, new_df], ignore_index=True)

    return merged_df