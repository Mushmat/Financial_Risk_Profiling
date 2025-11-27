import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from data_preprocessing import load_catboost_data
from config import SEED
import numpy as np

print("="*80)
print("CATBOOST CLASSIFIER")
print("="*80)

X_train, y_train, X_test, test_ids, _, cat_features = load_catboost_data()
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Using categorical columns at indices: {cat_features}")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
auc_scores = []

for train_idx, val_idx in cv.split(X_train, y_train):
    cb = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.12,
        l2_leaf_reg=7,
        loss_function='Logloss',
        verbose=0,
        random_state=SEED
    )
    cb.fit(
        X_train.iloc[train_idx], y_train.iloc[train_idx],
        cat_features=cat_features
    )
    val_preds = cb.predict_proba(X_train.iloc[val_idx])[:, 1]
    roc_auc = roc_auc_score(y_train.iloc[val_idx], val_preds)
    print(f"Fold ROC-AUC: {roc_auc:.4f}")
    auc_scores.append(roc_auc)

print(f"[CatBoost] Mean CV ROC-AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")

cb_final = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.12,
    l2_leaf_reg=7,
    loss_function='Logloss',
    verbose=0,
    random_state=SEED
)
cb_final.fit(X_train, y_train, cat_features=cat_features)
cb_preds = cb_final.predict(X_test)
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': cb_preds}).to_csv('catboost_submission.csv', index=False)
print("\nSubmission saved as catboost_submission.csv")
