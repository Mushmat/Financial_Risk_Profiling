import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from data_preprocessing import load_and_preprocess
from config import SEED
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("OPTIMIZED SVM CLASSIFIER - FULL DATASET")
print("="*80)

# Load data
X_train, y_train, X_test, test_ids, _ = load_and_preprocess()
print(f"\nDataset size: {X_train.shape[0]} samples, {X_train.shape[1]} features")

# ============================================================================
# PART 1: Fast LinearSVC (Primary Model)
# ============================================================================
print("\n" + "="*80)
print("TRAINING LINEAR SVM (FAST)")
print("="*80)

# Hyperparameter tuning for LinearSVC
print("\n[1] Hyperparameter tuning...")
param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
linear_svm_base = LinearSVC(max_iter=3000, random_state=SEED, dual=False)
grid_search = GridSearchCV(
    linear_svm_base, 
    param_grid, 
    cv=3, 
    scoring='roc_auc', 
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
best_C = grid_search.best_params_['C']
print(f"Best C parameter: {best_C}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# Train final LinearSVC with best C
print("\n[2] Training LinearSVC with best parameters...")
linear_svm = LinearSVC(C=best_C, max_iter=3000, random_state=SEED, dual=False)
linear_svm_calibrated = CalibratedClassifierCV(linear_svm, cv=5, method='sigmoid')
linear_svm_calibrated.fit(X_train, y_train)

# Cross-validation with calibrated model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
linear_svm_for_cv = LinearSVC(C=best_C, max_iter=3000, random_state=SEED, dual=False)
cv_scores_linear = cross_val_score(linear_svm_for_cv, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"\n[3] LinearSVC Cross-Validation ROC-AUC: {cv_scores_linear.mean():.4f} (+/- {cv_scores_linear.std():.4f})")

# Predict on test
print("\n[4] Predicting on test set (LinearSVC)...")
test_proba_linear = linear_svm_calibrated.predict_proba(X_test)[:, 1]
test_preds_linear = linear_svm_calibrated.predict(X_test)

# ============================================================================
# PART 2: RBF SVM on subset (for comparison)
# ============================================================================
print("\n" + "="*80)
print("TRAINING RBF SVM ON SUBSET (FOR COMPARISON)")
print("="*80)

# Use smaller subset for RBF (5000 samples)
subset_size = min(5000, len(X_train))
if subset_size < len(X_train):
    X_train_subset, _, y_train_subset, _ = train_test_split(
        X_train, y_train, train_size=subset_size, stratify=y_train, random_state=SEED
    )
    print(f"\n[5] Training RBF SVM on {subset_size} samples...")
else:
    X_train_subset, y_train_subset = X_train, y_train
    print(f"\n[5] Training RBF SVM on full dataset ({subset_size} samples)...")

# Train RBF SVM
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=SEED)
rbf_svm.fit(X_train_subset, y_train_subset)

# Cross-validation on subset
cv_subset = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
cv_scores_rbf = cross_val_score(rbf_svm, X_train_subset, y_train_subset, cv=cv_subset, scoring='roc_auc')
print(f"\n[6] RBF SVM Cross-Validation ROC-AUC (subset): {cv_scores_rbf.mean():.4f} (+/- {cv_scores_rbf.std():.4f})")

# Predict on test
print("\n[7] Predicting on test set (RBF SVM)...")
test_proba_rbf = rbf_svm.predict_proba(X_test)[:, 1]
test_preds_rbf = rbf_svm.predict(X_test)

# ============================================================================
# PART 3: Model Selection and Ensemble
# ============================================================================
print("\n" + "="*80)
print("MODEL SELECTION AND FINAL PREDICTION")
print("="*80)

# Choose best model or ensemble
if cv_scores_linear.mean() > cv_scores_rbf.mean():
    print("\n[8] LinearSVC selected as primary model (better CV score)")
    final_preds = test_preds_linear
    final_proba = test_proba_linear
    model_name = "linear"
else:
    print("\n[8] RBF SVM selected as primary model (better CV score)")
    final_preds = test_preds_rbf
    final_proba = test_proba_rbf
    model_name = "rbf"

# Optional: Ensemble both models (weighted average)
print("\n[9] Creating ensemble of both models...")
ensemble_proba = 0.7 * test_proba_linear + 0.3 * test_proba_rbf
ensemble_preds = (ensemble_proba > 0.5).astype(int)

# ============================================================================
# PART 4: Save Submissions
# ============================================================================
print("\n" + "="*80)
print("SAVING SUBMISSIONS")
print("="*80)

# Save individual model submissions
submission_linear = pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': test_preds_linear})
submission_linear.to_csv('svm_linear_submission.csv', index=False)
print("\n[10] LinearSVC submission saved as 'svm_linear_submission.csv'")

submission_rbf = pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': test_preds_rbf})
submission_rbf.to_csv('svm_rbf_submission.csv', index=False)
print("[11] RBF SVM submission saved as 'svm_rbf_submission.csv'")

# Save best model submission
submission_best = pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': final_preds})
submission_best.to_csv('svm_full_submission.csv', index=False)
print(f"[12] Best model ({model_name}) submission saved as 'svm_full_submission.csv'")

# Save ensemble submission
submission_ensemble = pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': ensemble_preds})
submission_ensemble.to_csv('svm_ensemble_submission.csv', index=False)
print("[13] Ensemble submission saved as 'svm_ensemble_submission.csv'")

# ============================================================================
# PART 5: Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nLinearSVC CV ROC-AUC: {cv_scores_linear.mean():.4f} (+/- {cv_scores_linear.std():.4f})")
print(f"RBF SVM CV ROC-AUC (subset): {cv_scores_rbf.mean():.4f} (+/- {cv_scores_rbf.std():.4f})")
print(f"\nBest Model: {model_name.upper()}")
print(f"Best C (LinearSVC): {best_C}")
print("\nFiles generated:")
print("  - svm_linear_submission.csv (LinearSVC)")
print("  - svm_rbf_submission.csv (RBF SVM)")
print("  - svm_full_submission.csv (Best model)")
print("  - svm_ensemble_submission.csv (Ensemble)")
print("="*80)
