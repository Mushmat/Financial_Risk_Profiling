import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from data_preprocessing import load_and_preprocess
from config import SEED
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("OPTIMIZED NEURAL NETWORK CLASSIFIER - FULL DATASET")
print("="*80)

# Load data
X_train, y_train, X_test, test_ids, _ = load_and_preprocess()
print(f"\nDataset size: {X_train.shape[0]} samples, {X_train.shape[1]} features")

# ============================================================================
# PART 1: Quick baseline NN
# ============================================================================
print("\n" + "="*80)
print("BASELINE NEURAL NETWORK")
print("="*80)

print("\n[1] Training baseline NN (2 layers)...")
nn_baseline = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=300,
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    verbose=False
)
nn_baseline.fit(X_train, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
baseline_scores = cross_val_score(nn_baseline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"[2] Baseline CV ROC-AUC: {baseline_scores.mean():.4f} (+/- {baseline_scores.std():.4f})")

# ============================================================================
# PART 2: Hyperparameter tuning (fast version)
# ============================================================================
print("\n" + "="*80)
print("HYPERPARAMETER TUNING")
print("="*80)

print("\n[3] Grid search for optimal parameters...")
param_grid = {
    'hidden_layer_sizes': [(100, 50), (80, 40), (64, 32)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

nn_grid = MLPClassifier(
    activation='relu',
    solver='adam',
    max_iter=300,
    early_stopping=True,
    random_state=SEED,
    verbose=False
)

grid_search = GridSearchCV(
    nn_grid,
    param_grid,
    cv=3,  # Only 3 folds for speed
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\n[4] Best parameters: {grid_search.best_params_}")
print(f"[5] Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# ============================================================================
# PART 3: Train optimized models
# ============================================================================
print("\n" + "="*80)
print("TRAINING OPTIMIZED MODELS")
print("="*80)

# Model 1: Optimized from GridSearch
print("\n[6] Training optimized NN with best params...")
nn_optimized = MLPClassifier(
    hidden_layer_sizes=grid_search.best_params_['hidden_layer_sizes'],
    alpha=grid_search.best_params_['alpha'],
    learning_rate_init=grid_search.best_params_['learning_rate_init'],
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=500,
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    verbose=False
)
nn_optimized.fit(X_train, y_train)

optimized_scores = cross_val_score(nn_optimized, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"[7] Optimized NN CV ROC-AUC: {optimized_scores.mean():.4f} (+/- {optimized_scores.std():.4f})")

# Model 2: Deeper network (for comparison)
print("\n[8] Training deeper NN (3 layers)...")
nn_deep = MLPClassifier(
    hidden_layer_sizes=(100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=400,
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    verbose=False
)
nn_deep.fit(X_train, y_train)

deep_scores = cross_val_score(nn_deep, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"[9] Deep NN CV ROC-AUC: {deep_scores.mean():.4f} (+/- {deep_scores.std():.4f})")

# ============================================================================
# PART 4: Select best model and create ensemble
# ============================================================================
print("\n" + "="*80)
print("MODEL SELECTION AND ENSEMBLE")
print("="*80)

# Compare all models
models = {
    'baseline': (nn_baseline, baseline_scores.mean()),
    'optimized': (nn_optimized, optimized_scores.mean()),
    'deep': (nn_deep, deep_scores.mean())
}

best_model_name = max(models, key=lambda x: models[x][1])
best_model = models[best_model_name][0]
best_score = models[best_model_name][1]

print(f"\n[10] Best model: {best_model_name.upper()} (CV ROC-AUC: {best_score:.4f})")

# ============================================================================
# PART 5: Predictions
# ============================================================================
print("\n[11] Generating predictions...")

# Individual predictions
pred_baseline = nn_baseline.predict(X_test)
pred_optimized = nn_optimized.predict(X_test)
pred_deep = nn_deep.predict(X_test)

# Probability predictions for ensemble
proba_baseline = nn_baseline.predict_proba(X_test)[:, 1]
proba_optimized = nn_optimized.predict_proba(X_test)[:, 1]
proba_deep = nn_deep.predict_proba(X_test)[:, 1]

# Weighted ensemble based on CV scores
total_score = baseline_scores.mean() + optimized_scores.mean() + deep_scores.mean()
w1 = baseline_scores.mean() / total_score
w2 = optimized_scores.mean() / total_score
w3 = deep_scores.mean() / total_score

ensemble_proba = w1 * proba_baseline + w2 * proba_optimized + w3 * proba_deep
ensemble_preds = (ensemble_proba > 0.5).astype(int)

# ============================================================================
# PART 6: Save submissions
# ============================================================================
print("\n[12] Saving submissions...")

# Individual models
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': pred_baseline}).to_csv('nn_baseline_submission.csv', index=False)
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': pred_optimized}).to_csv('nn_optimized_submission.csv', index=False)
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': pred_deep}).to_csv('nn_deep_submission.csv', index=False)

# Best single model
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': best_model.predict(X_test)}).to_csv('nn_full_submission.csv', index=False)

# Ensemble
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': ensemble_preds}).to_csv('nn_ensemble_submission.csv', index=False)

# ============================================================================
# PART 7: Summary
# ============================================================================
print("\n" + "="*80)
print("NEURAL NETWORK SUMMARY")
print("="*80)

print(f"\n📊 MODEL PERFORMANCE:")
print(f"  Baseline (64→32):     {baseline_scores.mean():.4f} (+/- {baseline_scores.std():.4f})")
print(f"  Optimized (tuned):    {optimized_scores.mean():.4f} (+/- {optimized_scores.std():.4f})")
print(f"  Deep (100→50→25):     {deep_scores.mean():.4f} (+/- {deep_scores.std():.4f})")

print(f"\n🏆 BEST MODEL: {best_model_name.upper()}")
print(f"  CV ROC-AUC: {best_score:.4f}")
print(f"  Architecture: {models[best_model_name][0].hidden_layer_sizes}")

print(f"\n⚖️  ENSEMBLE WEIGHTS:")
print(f"  Baseline:  {w1:.2%}")
print(f"  Optimized: {w2:.2%}")
print(f"  Deep:      {w3:.2%}")

print(f"\n📁 SUBMISSIONS SAVED:")
print(f"  - nn_baseline_submission.csv")
print(f"  - nn_optimized_submission.csv")
print(f"  - nn_deep_submission.csv")
print(f"  - nn_full_submission.csv (BEST SINGLE MODEL)")
print(f"  - nn_ensemble_submission.csv (RECOMMENDED)")

print(f"\n💡 TARGET: Beat 0.887")
if best_score > 0.887:
    print(f"  ✅ SUCCESS! Best model: {best_score:.4f} > 0.887")
elif ensemble_proba.mean() > 0.5:
    print(f"  ⚠️  Try nn_ensemble_submission.csv - may perform better on test set")
else:
    print(f"  📈 Current best: {best_score:.4f}. Try feature engineering or SVM+NN ensemble.")

print("="*80)
