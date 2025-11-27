import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
try:
    import modellightgbm as lgb
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False
from data_preprocessing import load_and_preprocess
from config import SEED
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TREE ENSEMBLE MODELS")
print("="*80)

# Load data
X_train, y_train, X_test, test_ids, _ = load_and_preprocess()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {}

# Random Forest
print("\n[1] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=SEED,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"Random Forest CV ROC-AUC: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
models['rf'] = (rf, rf_scores.mean())

# Extra Trees
print("\n[2] Training Extra Trees...")
et = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=SEED,
    n_jobs=-1
)
et.fit(X_train, y_train)
et_scores = cross_val_score(et, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"Extra Trees CV ROC-AUC: {et_scores.mean():.4f} (+/- {et_scores.std():.4f})")
models['et'] = (et, et_scores.mean())

# XGBoost
print("\n[3] Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"XGBoost CV ROC-AUC: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")
models['xgb'] = (xgb_model, xgb_scores.mean())

# LightGBM (if available)
if HAS_LIGHTGBM:
    print("\n[4] Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_scores = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"LightGBM CV ROC-AUC: {lgb_scores.mean():.4f} (+/- {lgb_scores.std():.4f})")
    models['lgb'] = (lgb_model, lgb_scores.mean())

# Gradient Boosting
print("\n[5] Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=SEED
)
gb.fit(X_train, y_train)
gb_scores = cross_val_score(gb, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"Gradient Boosting CV ROC-AUC: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")
models['gb'] = (gb, gb_scores.mean())

# Select best model
best_name = max(models, key=lambda x: models[x][1])
best_model = models[best_name][0]
print(f"\n[6] Best model: {best_name.upper()} with CV ROC-AUC: {models[best_name][1]:.4f}")

# Create weighted ensemble
print("\n[7] Creating weighted ensemble...")
test_probas = {}
for name, (model, score) in models.items():
    test_probas[name] = model.predict_proba(X_test)[:, 1]

# Weight by CV scores
total_score = sum(score for _, score in models.values())
weights = {name: score/total_score for name, (_, score) in models.items()}

ensemble_proba = sum(weights[name] * test_probas[name] for name in models.keys())
ensemble_preds = (ensemble_proba > 0.5).astype(int)

# Save submissions
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': rf.predict(X_test)}).to_csv('rf_submission.csv', index=False)
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': xgb_model.predict(X_test)}).to_csv('xgb_submission.csv', index=False)
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': best_model.predict(X_test)}).to_csv('best_tree_submission.csv', index=False)
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': ensemble_preds}).to_csv('tree_ensemble_submission.csv', index=False)

print("\n[8] Submissions saved:")
print("  - rf_submission.csv (Random Forest)")
print("  - xgb_submission.csv (XGBoost)")
print("  - best_tree_submission.csv (Best single tree model)")
print("  - tree_ensemble_submission.csv (Weighted ensemble)")
print("="*80)
