import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from config import TRAIN_PATH, TARGET, ID_COL, SEED

def add_features(df):
    df = df.copy()
    # Ratios and interactions
    df['RequestIncomeRatio'] = df['RequestedSum'] / (df['AnnualEarnings'] + 1)
    df['DebtIncomeRatio'] = df['DebtFactor'] * df['AnnualEarnings']
    df['AccountsPerYearWork'] = df['ActiveAccounts'] / (df['WorkDuration'] + 1)
    df['OfferPeriodProduct'] = df['OfferRate'] * df['RepayPeriod']
    df['TrustWorkRatio'] = df['TrustMetric'] / (df['WorkDuration'] + 1)
    # Non-linear transforms
    df['LogAnnualEarnings'] = np.log1p(df['AnnualEarnings'])
    df['LogRequestedSum'] = np.log1p(df['RequestedSum'])
    df['LogDebtIncomeRatio'] = np.log1p(df['DebtIncomeRatio'].clip(lower=0))
    # Boolean flags
    df['HighDebt'] = (df['DebtFactor'] > 0.5).astype(int)
    df['VeryHighDebt'] = (df['DebtFactor'] > 0.75).astype(int)
    df['YoungApplicant'] = (df['ApplicantYears'] < 25).astype(int)
    df['SeniorApplicant'] = (df['ApplicantYears'] > 55).astype(int)
    df['ShortRepay'] = (df['RepayPeriod'] < 18).astype(int)
    df['LongRepay'] = (df['RepayPeriod'] > 36).astype(int)
    if 'OwnsProperty' in df.columns:
        df['HighIncomeHasProperty'] = ((df['AnnualEarnings'] > 120000) & (df['OwnsProperty'] == 1)).astype(int)
    else:
        df['HighIncomeHasProperty'] = 0
    df['WorkDebtInteraction'] = df['WorkDuration'] * df['DebtFactor']
    df['AccountsDebtInteraction'] = df['ActiveAccounts'] * df['DebtFactor']
    df['TrustDebtInteraction'] = df['TrustMetric'] * df['DebtFactor']
    return df

def load_train_split(scale=True):
    """Load Kaggle train.csv, do feature engineering, split into train/test for viva."""
    train = pd.read_csv(TRAIN_PATH)
    y = train[TARGET]
    ids = train[ID_COL]
    X = train.drop([ID_COL, TARGET], axis=1)

    # Feature engineering
    X = add_features(X)

    # Train/test split for evaluation
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=SEED, stratify=y
    )

    # Identify categorical columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Label encode categoricals using train+test values
    for col in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
        le.fit(all_vals)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    # Fill numeric NAs
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, id_train.values, id_test.values
    else:
        return X_train.values, X_test.values, y_train.values, y_test.values, id_train.values, id_test.values

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, id_tr, id_te = load_train_split()
    print("Train shape:", X_tr.shape, "Test shape:", X_te.shape)
    print("Train target distribution:\n", pd.Series(y_tr).value_counts())
    print("Test target distribution:\n", pd.Series(y_te).value_counts())
