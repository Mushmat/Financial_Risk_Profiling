import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from data_preprocessing import load_and_preprocess
from config import SEED
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FAST ENSEMBLE: LinearSVM + Neural Network")
print("="*80)

# Load data
X_train, y_train, X_test, test_ids, _ = load_and_preprocess()
print(f"\nDataset size: {X_train.shape[0]} samples, {X_train.shape[1]} features")

# ============================================================================
# Define fast models
# ============================================================================
print("\n[1] Defining models...")

# Fast LinearSVC (100x faster than RBF SVM)
svm_base = LinearSVC(C=1.0, max_iter=2000, random_state=SEED, dual=False)
svm = CalibratedClassifierCV(svm_base, cv=3, method='sigmoid')

# Faster Neural Network (smaller, early stopping)
nn = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # Smaller network
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=300,  # Fewer iterations
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    verbose=False
)

# ============================================================================
# Train individual models
# ============================================================================
print("\n[2] Training individual models...")
print("  Training LinearSVC...")
svm.fit(X_train, y_train)
print("  ✓ LinearSVC trained")

print("  Training Neural Network...")
nn.fit(X_train, y_train)
print("  ✓ Neural Network trained")

# Quick CV scores
print("\n[3] Evaluating individual models...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

svm_for_cv = LinearSVC(C=1.0, max_iter=2000, random_state=SEED, dual=False)
svm_scores = cross_val_score(svm_for_cv, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"  LinearSVC CV ROC-AUC: {svm_scores.mean():.4f} (+/- {svm_scores.std():.4f})")

nn_for_cv = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=SEED, early_stopping=True)
nn_scores = cross_val_score(nn_for_cv, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"  Neural Network CV ROC-AUC: {nn_scores.mean():.4f} (+/- {nn_scores.std():.4f})")

# ============================================================================
# Create ensemble predictions
# ============================================================================
print("\n[4] Creating ensemble predictions...")

# Predict probabilities
svm_proba = svm.predict_proba(X_test)[:, 1]
nn_proba = nn.predict_proba(X_test)[:, 1]

# Weighted average (adjust weights based on CV scores)
svm_weight = svm_scores.mean()
nn_weight = nn_scores.mean()
total_weight = svm_weight + nn_weight

ensemble_proba = (svm_weight * svm_proba + nn_weight * nn_proba) / total_weight
ensemble_preds = (ensemble_proba > 0.5).astype(int)

# Also save individual predictions
svm_preds = svm.predict(X_test)
nn_preds = nn.predict(X_test)

# Simple majority vote ensemble (alternative)
vote_preds = ((svm_preds.astype(int) + nn_preds.astype(int)) > 1).astype(int)

# ============================================================================
# Save submissions
# ============================================================================
print("\n[5] Saving submissions...")

# Individual model submissions
submission_svm = pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': svm_preds})
submission_svm.to_csv('ensemble_svm_only.csv', index=False)

submission_nn = pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': nn_preds})
submission_nn.to_csv('ensemble_nn_only.csv', index=False)

# Weighted ensemble
submission_weighted = pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': ensemble_preds})
submission_weighted.to_csv('ensemble_weighted_submission.csv', index=False)

# Majority vote ensemble
submission_vote = pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': vote_preds})
submission_vote.to_csv('ensemble_vote_submission.csv', index=False)

# Best ensemble (weighted is usually better)
submission_best = pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': ensemble_preds})
submission_best.to_csv('ensemble_submission.csv', index=False)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("ENSEMBLE SUMMARY")
print("="*80)

print(f"\n📊 INDIVIDUAL MODEL PERFORMANCE:")
print(f"  LinearSVC CV ROC-AUC:     {svm_scores.mean():.4f}")
print(f"  Neural Network CV ROC-AUC: {nn_scores.mean():.4f}")

print(f"\n⚖️  ENSEMBLE WEIGHTS:")
print(f"  LinearSVC:     {svm_weight/(svm_weight+nn_weight):.2%}")
print(f"  Neural Network: {nn_weight/(svm_weight+nn_weight):.2%}")

print(f"\n📁 SUBMISSIONS SAVED:")
print(f"  - ensemble_svm_only.csv (LinearSVC only)")
print(f"  - ensemble_nn_only.csv (Neural Network only)")
print(f"  - ensemble_weighted_submission.csv (Weighted average)")
print(f"  - ensemble_vote_submission.csv (Majority vote)")
print(f"  - ensemble_submission.csv (Best ensemble)")

print("\n💡 RECOMMENDATION:")
if svm_scores.mean() > nn_scores.mean() + 0.01:
    print("  → LinearSVC significantly outperforms NN. Try SVM-only submission first.")
elif nn_scores.mean() > svm_scores.mean() + 0.01:
    print("  → Neural Network outperforms LinearSVC. Try NN-only submission first.")
else:
    print("  → Models perform similarly. Ensemble likely to perform best.")

print("="*80)
