import pandas as pd
import numpy as np
from lightGBM import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from data_preprocessing import load_and_preprocess
from config import SEED
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LIGHTGBM CLASSIFIER")
print("="*80)

X_train, y_train, X_test, test_ids, _ = load_and_preprocess()
# Optionally add your engineered features here
# X_train = add_features(X_train)
# X_test = add_features(X_test)

lgb = LGBMClassifier(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.09,
    num_leaves=32,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=8,
    reg_alpha=2,
    random_state=SEED,
    n_jobs=-1
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_score = cross_val_score(lgb, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"[LightGBM] CV ROC-AUC: {cv_score.mean():.4f} (+/- {cv_score.std():.4f})")

lgb.fit(X_train, y_train)
lgb_preds = lgb.predict(X_test)
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': lgb_preds}).to_csv('lgbm_submission.csv', index=False)
print("\nSubmission saved as lgbm_submission.csv")
