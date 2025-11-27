import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from config import TRAIN_PATH, TEST_PATH, TARGET, ID_COL, SEED

def add_features(df):
    """Advanced feature engineering with financial risk indicators."""
    df = df.copy()
    
    # ======================
    # BASIC RATIOS
    # ======================
    df['RequestIncomeRatio'] = df['RequestedSum'] / (df['AnnualEarnings'] + 1)
    df['DebtIncomeRatio'] = df['DebtFactor'] * df['AnnualEarnings']
    df['AccountsPerYearWork'] = df['ActiveAccounts'] / (df['WorkDuration'] + 1)
    df['OfferPeriodProduct'] = df['OfferRate'] * df['RepayPeriod']
    df['TrustWorkRatio'] = df['TrustMetric'] / (df['WorkDuration'] + 1)
    
    # ======================
    # ADVANCED FINANCIAL RATIOS
    # ======================
    # Payment capacity indicators
    df['MonthlyIncome'] = df['AnnualEarnings'] / 12
    df['EstimatedMonthlyPayment'] = df['RequestedSum'] / (df['RepayPeriod'] + 1)
    df['PaymentToIncomeRatio'] = df['EstimatedMonthlyPayment'] / (df['MonthlyIncome'] + 1)
    df['RemainingIncomeAfterPayment'] = df['MonthlyIncome'] - df['EstimatedMonthlyPayment']
    
    # Debt burden metrics
    df['TotalDebtAmount'] = df['DebtFactor'] * df['RequestedSum']
    df['DebtPerAccount'] = df['TotalDebtAmount'] / (df['ActiveAccounts'] + 1)
    df['RequestPerAccount'] = df['RequestedSum'] / (df['ActiveAccounts'] + 1)
    
    # Age and experience factors
    df['AgeWorkRatio'] = df['ApplicantYears'] / (df['WorkDuration'] + 1)
    df['WorkExperienceRatio'] = df['WorkDuration'] / (df['ApplicantYears'] + 1)
    df['YearsBeforeWork'] = df['ApplicantYears'] - df['WorkDuration']
    
    # Trust and risk combinations
    df['TrustAgeProduct'] = df['TrustMetric'] * df['ApplicantYears']
    df['TrustIncomeProduct'] = df['TrustMetric'] * df['AnnualEarnings'] / 100000
    df['RiskScore'] = df['DebtFactor'] * df['RequestIncomeRatio'] * (1 / (df['TrustMetric'] + 0.1))
    
    # Loan characteristics
    df['OfferRatePerYear'] = df['OfferRate'] / (df['RepayPeriod'] / 12 + 1)
    df['TotalRepaymentAmount'] = df['RequestedSum'] * (1 + df['OfferRate'] * df['RepayPeriod'] / 1200)
    df['InterestAmount'] = df['TotalRepaymentAmount'] - df['RequestedSum']
    df['InterestToRequestRatio'] = df['InterestAmount'] / (df['RequestedSum'] + 1)
    
    # ======================
    # POLYNOMIAL FEATURES (selective)
    # ======================
    df['DebtFactor_Squared'] = df['DebtFactor'] ** 2
    df['TrustMetric_Squared'] = df['TrustMetric'] ** 2
    df['RequestIncomeRatio_Squared'] = df['RequestIncomeRatio'] ** 2
    df['OfferRate_Squared'] = df['OfferRate'] ** 2
    
    # Log transforms for skewed features
    df['Log_RequestedSum'] = np.log1p(df['RequestedSum'])
    df['Log_AnnualEarnings'] = np.log1p(df['AnnualEarnings'])
    df['Log_ActiveAccounts'] = np.log1p(df['ActiveAccounts'])
    
    # ======================
    # BOOLEAN FLAGS (Risk Indicators)
    # ======================
    df['HighDebt'] = (df['DebtFactor'] > 0.5).astype(int)
    df['VeryHighDebt'] = (df['DebtFactor'] > 0.7).astype(int)
    df['YoungApplicant'] = (df['ApplicantYears'] < 25).astype(int)
    df['ShortRepay'] = (df['RepayPeriod'] < 18).astype(int)
    df['LongRepay'] = (df['RepayPeriod'] > 48).astype(int)
    df['HighInterestRate'] = (df['OfferRate'] > 15).astype(int)
    df['LowTrust'] = (df['TrustMetric'] < 0.3).astype(int)
    df['HighTrust'] = (df['TrustMetric'] > 0.7).astype(int)
    df['ShortWorkHistory'] = (df['WorkDuration'] < 2).astype(int)
    df['LongWorkHistory'] = (df['WorkDuration'] > 10).astype(int)
    df['LowIncome'] = (df['AnnualEarnings'] < 50000).astype(int)
    df['HighIncome'] = (df['AnnualEarnings'] > 150000).astype(int)
    df['LargeRequest'] = (df['RequestedSum'] > 100000).astype(int)
    df['ManyAccounts'] = (df['ActiveAccounts'] > 5).astype(int)
    df['FewAccounts'] = (df['ActiveAccounts'] <= 1).astype(int)
    
    # Combined risk flags
    df['HighRiskProfile'] = (
        (df['DebtFactor'] > 0.6) & 
        (df['TrustMetric'] < 0.4) & 
        (df['RequestIncomeRatio'] > 2)
    ).astype(int)
    
    df['LowRiskProfile'] = (
        (df['DebtFactor'] < 0.3) & 
        (df['TrustMetric'] > 0.6) & 
        (df['WorkDuration'] > 5) &
        (df['PaymentToIncomeRatio'] < 0.3)
    ).astype(int)
    
    df['UnstableFinances'] = (
        (df['WorkDuration'] < 2) & 
        (df['ActiveAccounts'] > 4) &
        (df['DebtFactor'] > 0.4)
    ).astype(int)
    
    # Property ownership interactions
    if 'OwnsProperty' in df.columns:
        df['HighIncomeHasProperty'] = ((df['AnnualEarnings'] > 120000) & (df['OwnsProperty'] == 1)).astype(int)
        df['PropertyNoDebt'] = ((df['OwnsProperty'] == 1) & (df['DebtFactor'] < 0.3)).astype(int)
        df['NoPropertyHighDebt'] = ((df['OwnsProperty'] == 0) & (df['DebtFactor'] > 0.5)).astype(int)
        df['PropertyTrustProduct'] = df['OwnsProperty'] * df['TrustMetric']
    else:
        df['HighIncomeHasProperty'] = 0
        df['PropertyNoDebt'] = 0
        df['NoPropertyHighDebt'] = 0
        df['PropertyTrustProduct'] = 0
    
    # ======================
    # BINNED FEATURES
    # ======================
    df['Age_Bin'] = pd.cut(df['ApplicantYears'], bins=[0, 25, 35, 45, 100], labels=[0, 1, 2, 3])
    df['Income_Bin'] = pd.cut(df['AnnualEarnings'], bins=[0, 50000, 100000, 150000, 1e9], labels=[0, 1, 2, 3])
    df['Request_Bin'] = pd.cut(df['RequestedSum'], bins=[0, 50000, 100000, 200000, 1e9], labels=[0, 1, 2, 3])
    df['Debt_Bin'] = pd.cut(df['DebtFactor'], bins=[0, 0.25, 0.5, 0.75, 1.0], labels=[0, 1, 2, 3])
    
    # Convert bins to numeric
    for col in ['Age_Bin', 'Income_Bin', 'Request_Bin', 'Debt_Bin']:
        df[col] = df[col].astype(float)
    
    return df

def load_catboost_data():
    """Load data for CatBoost (maintains categorical columns)."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y_train = train[TARGET]
    train_ids = train[ID_COL]
    test_ids = test[ID_COL]
    X_train = train.drop([ID_COL, TARGET], axis=1)
    X_test = test.drop([ID_COL], axis=1)
    
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    cat_features = [X_train.columns.get_loc(col) for col in cat_cols]
    return X_train, y_train, X_test, test_ids, train_ids, cat_features

def load_and_preprocess(use_robust_scaler=False):
    """
    Load, feature engineer, encode and scale data.
    
    Args:
        use_robust_scaler: If True, use RobustScaler instead of StandardScaler
                          (better for outliers)
    """
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # Separate target and IDs
    y_train = train[TARGET]
    train_ids = train[ID_COL]
    test_ids = test[ID_COL]

    # Drop ID and target
    X_train = train.drop([ID_COL, TARGET], axis=1)
    X_test = test.drop([ID_COL], axis=1)

    # --- FEATURE ENGINEERING ---
    print("Applying feature engineering...")
    X_train = add_features(X_train)
    X_test = add_features(X_test)
    print(f"After feature engineering: {X_train.shape[1]} features")

    # Identify categorical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {cat_cols}")

    # Label encode categorical features
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    # Fill missing values if any (after creating new features)
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    # Replace infinities with large finite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    # Scale features
    if use_robust_scaler:
        scaler = RobustScaler()  # More robust to outliers
    else:
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, test_ids, train_ids

if __name__ == "__main__":
    X_train, y_train, X_test, test_ids, train_ids = load_and_preprocess()
    print(f"\nPreprocessed train shape: {X_train.shape}")
    print(f"Preprocessed test shape: {X_test.shape}")
    print(f"Target distribution:\n{pd.Series(y_train).value_counts()}")
    print(f"Target distribution (%):\n{pd.Series(y_train).value_counts(normalize=True) * 100}")