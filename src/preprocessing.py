import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Provera nedostajucih vrednosti
    # print(df.isnull().sum())
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

    df['School_Grade'] = df['School_Grade'].astype(str).str.replace(r'th|st|nd|rd', '', regex=True)

    # Pretvoriš u numerik, a ako nešto ne može da se pretvori, postane NaN (umesto da pukne kod)
    df['School_Grade'] = pd.to_numeric(df['School_Grade'], errors='coerce')

    # Tek onda popuniš NaN (ako ih ima) i pretvoriš u int
    df['School_Grade'] = df['School_Grade'].fillna(0).astype(int)

    # Izbacujemo Location jer ima previse unique vrijednosti, pa bismo dobili previse novih kolona, a i nema nam uticaja
    df = df.drop(columns=['Location'])

    # One-Hot encoding za kategroicke vrijednosti
    categorical_cols = ['Gender', 'Phone_Usage_Purpose']

    #je ključan da ne dupliramo informacije (npr. ako znamo da nije Muško i nije Ostalo, onda je Žensko)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df_encoded