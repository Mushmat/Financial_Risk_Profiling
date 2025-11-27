import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from data_preprocessing import load_and_preprocess
from config import SEED
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NEURAL NETWORK CLASSIFIER - 20% SUBSET")
print("="*80)

# Load data
X_train_full, y_train_full, X_test, test_ids, _ = load_and_preprocess()

# Create 20% subset
X_train_small, _, y_train_small, _ = train_test_split(
    X_train_full, y_train_full, train_size=0.2, stratify=y_train_full, random_state=SEED
)

print(f"\n[1] Training on {len(X_train_small)} samples (20% of full data)")

# Train Neural Network
nn_small = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=300,
    random_state=SEED,
    early_stopping=True
)
nn_small.fit(X_train_small, y_train_small)

# Cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(nn_small, X_train_small, y_train_small, cv=cv, scoring='roc_auc')
print(f"\n[2] Cross-Validation ROC-AUC (20% subset): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Predict on test
test_preds = nn_small.predict(X_test)

# Save submission
submission = pd.DataFrame({
    'ProfileID': test_ids,
    'RiskFlag': test_preds
})
submission.to_csv('nn_20percent_submission.csv', index=False)
print("\n[3] Submission saved as 'nn_20percent_submission.csv'")
print("="*80)
