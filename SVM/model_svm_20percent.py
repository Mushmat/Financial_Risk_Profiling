import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from data_preprocessing import load_and_preprocess
from config import SEED
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FAST SVM CLASSIFIER - 20% SUBSET")
print("="*80)

# Load data
X_train_full, y_train_full, X_test, test_ids, _ = load_and_preprocess()
print(f"\nFull dataset size: {X_train_full.shape[0]} samples")

# Create 20% subset
X_train_small, _, y_train_small, _ = train_test_split(
    X_train_full, y_train_full, train_size=0.2, stratify=y_train_full, random_state=SEED
)

print(f"20% subset size: {len(X_train_small)} samples ({len(X_train_small)/len(X_train_full)*100:.1f}%)")

# ============================================================================
# Train LinearSVC on 20% subset (no hyperparameter tuning for speed)
# ============================================================================
print("\n[1] Training LinearSVC on 20% subset...")
linear_svm = LinearSVC(C=1.0, max_iter=2000, random_state=SEED, dual=False)
linear_svm_calibrated = CalibratedClassifierCV(linear_svm, cv=3, method='sigmoid')
linear_svm_calibrated.fit(X_train_small, y_train_small)

# Quick cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
linear_svm_cv = LinearSVC(C=1.0, max_iter=2000, random_state=SEED, dual=False)
cv_scores_20 = cross_val_score(linear_svm_cv, X_train_small, y_train_small, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"[2] LinearSVC CV ROC-AUC (20% subset): {cv_scores_20.mean():.4f} (+/- {cv_scores_20.std():.4f})")

# Predict on test
test_preds_20 = linear_svm_calibrated.predict(X_test)

# ============================================================================
# Quick comparison with full dataset (simple, no GridSearch)
# ============================================================================
print("\n[3] Quick comparison with full dataset...")
full_svm = LinearSVC(C=1.0, max_iter=2000, random_state=SEED, dual=False)
cv_full = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)  # Only 3 folds for speed
cv_scores_100 = cross_val_score(full_svm, X_train_full, y_train_full, cv=cv_full, scoring='roc_auc', n_jobs=-1)
print(f"[4] LinearSVC CV ROC-AUC (100% dataset): {cv_scores_100.mean():.4f} (+/- {cv_scores_100.std():.4f})")

# Train on full and predict
full_svm_cal = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=SEED, dual=False), cv=3)
full_svm_cal.fit(X_train_full, y_train_full)
test_preds_100 = full_svm_cal.predict(X_test)

# ============================================================================
# Save submissions
# ============================================================================
print("\n[5] Saving submissions...")
submission_20 = pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': test_preds_20})
submission_20.to_csv('svm_20percent_submission.csv', index=False)

submission_100 = pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': test_preds_100})
submission_100.to_csv('svm_100percent_submission.csv', index=False)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
performance_drop = cv_scores_100.mean() - cv_scores_20.mean()
print(f"\n20% Subset CV ROC-AUC:  {cv_scores_20.mean():.4f}")
print(f"100% Dataset CV ROC-AUC: {cv_scores_100.mean():.4f}")
print(f"Performance drop:        {performance_drop:.4f} ({abs(performance_drop)/cv_scores_100.mean()*100:.1f}%)")

print("\n📁 Submissions saved:")
print("  - svm_20percent_submission.csv")
print("  - svm_100percent_submission.csv")
print("="*80)
