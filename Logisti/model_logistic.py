import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from data_preprocessing import load_and_preprocess
from config import SEED
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LOGISTIC REGRESSION CLASSIFIER")
print("="*80)

# Load data
X_train, y_train, X_test, test_ids, _ = load_and_preprocess()

# Hyperparameter tuning
print("\n[1] Hyperparameter tuning...")
param_grid = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

lr = LogisticRegression(max_iter=1000, random_state=SEED)
grid = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"\n[2] Best parameters: {grid.best_params_}")
print(f"[3] Best CV ROC-AUC: {grid.best_score_:.4f}")

# Train final model
best_lr = LogisticRegression(
    C=grid.best_params_['C'],
    penalty=grid.best_params_['penalty'],
    solver='liblinear',
    max_iter=1000,
    random_state=SEED
)
best_lr.fit(X_train, y_train)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(best_lr, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"\n[4] Final CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Predict
test_preds = best_lr.predict(X_test)
test_proba = best_lr.predict_proba(X_test)[:, 1]

# Save
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': test_preds}).to_csv('logistic_submission.csv', index=False)
print("\n[5] Submission saved as 'logistic_submission.csv'")
print("="*80)
