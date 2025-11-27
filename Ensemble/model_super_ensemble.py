import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
from data_preprocessing import load_and_preprocess
from config import SEED
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SUPER ENSEMBLE - ALL MODELS COMBINED")
print("="*80)

X_train, y_train, X_test, test_ids, _ = load_and_preprocess()

# Define diverse models
models = {
    'LogisticRegression': LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
    'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=12, random_state=SEED, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=SEED, eval_metric='logloss', use_label_encoder=False),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=SEED),
    'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=SEED),
}

# Add LinearSVC (needs calibration)
svm_base = LinearSVC(C=1.0, max_iter=2000, random_state=SEED, dual=False)
models['SVM'] = CalibratedClassifierCV(svm_base, cv=3)

print("\n[1] Training all models and collecting predictions...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
test_probas = {}
cv_scores_dict = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)
    
    # CV score
    if name == 'SVM':
        cv_model = LinearSVC(C=1.0, max_iter=2000, random_state=SEED, dual=False)
        cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    cv_scores_dict[name] = cv_scores.mean()
    print(f"  {name} CV ROC-AUC: {cv_scores.mean():.4f}")
    
    # Test predictions
    test_probas[name] = model.predict_proba(X_test)[:, 1]

# Weighted ensemble by CV performance
print("\n[2] Creating weighted super-ensemble...")
total_score = sum(cv_scores_dict.values()) 
weights = {name: score/total_score for name, score in cv_scores_dict.items()}

super_ensemble_proba = sum(weights[name] * test_probas[name] for name in models.keys())
super_ensemble_preds = (super_ensemble_proba > 0.5).astype(int)

# Also try simple average
simple_avg_proba = np.mean(list(test_probas.values()), axis=0)
simple_avg_preds = (simple_avg_proba > 0.5).astype(int)

# Save submissions
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': super_ensemble_preds}).to_csv('super_ensemble_weighted.csv', index=False)
pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': simple_avg_preds}).to_csv('super_ensemble_avg.csv', index=False)

print("\n" + "="*80)
print("SUPER ENSEMBLE SUMMARY")
print("="*80)
print("\n📊 Individual Model Performance:")
for name, score in sorted(cv_scores_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {score:.4f} (weight: {weights[name]:.2%})")

print("\n📁 Submissions:")
print("  - super_ensemble_weighted.csv (RECOMMENDED)")
print("  - super_ensemble_avg.csv")
print("="*80)
