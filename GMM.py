import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from data_preprocessing import load_and_preprocess
from config import SEED
import numpy as np

print("="*80)
print("GAUSSIAN MIXTURE MODEL (GMM)")
print("="*80)

X_train, y_train, X_test, test_ids, _ = load_and_preprocess()
print(f"Train shape: {X_train.shape}")

# Fit two GMMs, one per class
gmm0 = GaussianMixture(n_components=3, covariance_type='full', random_state=SEED)
gmm1 = GaussianMixture(n_components=3, covariance_type='full', random_state=SEED)

X0 = X_train[y_train == 0]
X1 = X_train[y_train == 1]

gmm0.fit(X0)
gmm1.fit(X1)

# Estimate log-likelihood ratio for classification
ll0 = gmm0.score_samples(X_test)
ll1 = gmm1.score_samples(X_test)
proba = 1 / (1 + np.exp(ll0 - ll1))
preds = (proba > 0.5).astype(int)

pd.DataFrame({'ProfileID': test_ids, 'RiskFlag': preds}).to_csv('gmm_submission.csv', index=False)
print("Submission saved as gmm_submission.csv")
