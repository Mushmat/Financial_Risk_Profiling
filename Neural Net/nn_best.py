import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from data_preprocessing import load_and_preprocess
from config import SEED
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ENHANCED NN ENSEMBLE - ADVANCED ARCHITECTURE")
print("="*80)

# Load data with robust scaling
X_train, y_train, X_test, test_ids, _ = load_and_preprocess(use_robust_scaler=True)
print(f"Train size: {X_train.shape}")
print(f"Target distribution: {np.bincount(y_train)} (0s and 1s)")
print(f"Positive class ratio: {y_train.mean():.4f}")

# Calculate class weights for imbalanced data
class_weights = {0: 1.0, 1: (1 - y_train.mean()) / y_train.mean()}
print(f"Class weights: {class_weights}")

# ====================================================================================
# DIVERSE NEURAL NETWORK ARCHITECTURES
# ====================================================================================
# Strategy: Mix shallow, medium, and deeper networks with different configurations
# to capture different patterns in the data

settings = [
    # Architecture: (hidden_layers, seed, alpha, activation, learning_rate, solver)
    
    # Shallow networks - fast learners
    ((64, 32), SEED, 0.0001, 'relu', 0.001, 'adam'),
    ((48, 24), SEED+1, 0.0005, 'relu', 0.002, 'adam'),
    ((80, 40), SEED+2, 0.0002, 'tanh', 0.001, 'adam'),
    
    # Medium depth networks - balanced
    ((96, 48, 24), SEED+3, 0.0001, 'relu', 0.001, 'adam'),
    ((80, 60, 30), SEED+4, 0.0003, 'relu', 0.002, 'adam'),
    ((64, 48, 24), SEED+5, 0.0002, 'tanh', 0.001, 'adam'),
    
    # Wider networks - more capacity
    ((128, 64), SEED+6, 0.0001, 'relu', 0.001, 'adam'),
    ((100, 50), SEED+7, 0.0005, 'relu', 0.002, 'adam'),
    
    # Deeper networks - complex patterns
    ((80, 60, 40, 20), SEED+8, 0.0001, 'relu', 0.0015, 'adam'),
    ((64, 48, 32, 16), SEED+9, 0.0002, 'relu', 0.001, 'adam'),
    
    # Alternative activations
    ((72, 36), SEED+10, 0.0003, 'tanh', 0.001, 'adam'),
    ((64, 32, 16), SEED+11, 0.0002, 'logistic', 0.002, 'adam'),
    
    # Pyramid architectures
    ((100, 75, 50, 25), SEED+12, 0.0001, 'relu', 0.001, 'adam'),
    ((90, 60, 30), SEED+13, 0.0005, 'relu', 0.002, 'adam'),
    
    # High capacity models
    ((120, 80, 40), SEED+14, 0.0001, 'relu', 0.001, 'adam'),
]

print(f"\nTraining {len(settings)} diverse neural networks...")
print("-"*80)

test_probas = []
oof_predictions = np.zeros(len(y_train))  # Out-of-fold predictions
oof_counts = np.zeros(len(y_train))  # Count how many times each sample was predicted

model_weights = []  # Store model weights based on performance

for idx, (hls, s, alpha, act, lr, solver) in enumerate(settings, 1):
    print(f"\n[{idx}/{len(settings)}] Training NN: hidden_layers={hls}, seed={s}")
    print(f"  Params: alpha={alpha}, activation={act}, lr={lr}")
    
    # Create neural network with optimized parameters
    nn = MLPClassifier(
        hidden_layer_sizes=hls,
        activation=act,
        solver=solver,
        alpha=alpha,
        max_iter=300,  # Increased iterations
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,  # More patience
        learning_rate_init=lr,
        learning_rate='adaptive',
        random_state=s,
        verbose=False,
        batch_size='auto',
        shuffle=True,
        tol=1e-4,
        momentum=0.9,
        nesterovs_momentum=True,
        beta_1=0.9,
        beta_2=0.999
    )
    
    # Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
    fold_scores = []
    fold_auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train the model
        nn.fit(X_tr, y_tr)
        
        # Predict on validation set
        val_proba = nn.predict_proba(X_val)[:, 1]
        val_pred = (val_proba > 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_val, val_pred)
        auc = roc_auc_score(y_val, val_proba)
        
        fold_scores.append(acc)
        fold_auc_scores.append(auc)
        
        # Store out-of-fold predictions
        oof_predictions[val_idx] += val_proba
        oof_counts[val_idx] += 1
    
    mean_acc = np.mean(fold_scores)
    mean_auc = np.mean(fold_auc_scores)
    std_auc = np.std(fold_auc_scores)
    
    print(f"  CV Results: Accuracy={mean_acc:.4f}, AUC={mean_auc:.4f} (+/- {std_auc:.4f})")
    
    # Train on full training data
    nn.fit(X_train, y_train)
    
    # Predict on test set
    test_proba = nn.predict_proba(X_test)[:, 1]
    test_probas.append(test_proba)
    
    # Weight models by their AUC performance
    model_weights.append(mean_auc)

print("\n" + "="*80)
print("ENSEMBLE RESULTS")
print("="*80)

# Calculate out-of-fold ensemble performance
oof_predictions = oof_predictions / oof_counts
oof_pred_binary = (oof_predictions > 0.5).astype(int)
oof_acc = accuracy_score(y_train, oof_pred_binary)
oof_auc = roc_auc_score(y_train, oof_predictions)

print(f"\nOut-of-Fold Ensemble Performance:")
print(f"  Accuracy: {oof_acc:.4f}")
print(f"  AUC: {oof_auc:.4f}")

# ====================================================================================
# WEIGHTED ENSEMBLE
# ====================================================================================
# Normalize weights
model_weights = np.array(model_weights)
model_weights = model_weights / model_weights.sum()

print(f"\nModel weights (top 5):")
top_indices = np.argsort(model_weights)[-5:][::-1]
for i in top_indices:
    print(f"  Model {i+1}: {model_weights[i]:.4f}")

# Simple average ensemble
ensemble_proba_avg = np.mean(test_probas, axis=0)
ensemble_preds_avg = (ensemble_proba_avg > 0.5).astype(int)

# Weighted average ensemble
test_probas_array = np.array(test_probas)
ensemble_proba_weighted = np.average(test_probas_array, axis=0, weights=model_weights)
ensemble_preds_weighted = (ensemble_proba_weighted > 0.5).astype(int)

# ====================================================================================
# THRESHOLD OPTIMIZATION
# ====================================================================================
# Find optimal threshold based on OOF predictions
thresholds = np.linspace(0.3, 0.7, 41)
best_threshold = 0.5
best_oof_acc = 0

for thresh in thresholds:
    oof_pred_thresh = (oof_predictions > thresh).astype(int)
    acc = accuracy_score(y_train, oof_pred_thresh)
    if acc > best_oof_acc:
        best_oof_acc = acc
        best_threshold = thresh

print(f"\nOptimal threshold: {best_threshold:.3f} (OOF Accuracy: {best_oof_acc:.4f})")

# Apply optimal threshold
ensemble_preds_optimized = (ensemble_proba_weighted > best_threshold).astype(int)

# ====================================================================================
# SAVE SUBMISSIONS
# ====================================================================================
# Save weighted ensemble with optimized threshold (best approach)
submission_best = pd.DataFrame({
    'ProfileID': test_ids, 
    'RiskFlag': ensemble_preds_optimized
})
submission_best.to_csv('nn_leaderboard_submission.csv', index=False)

# Also save simple average for comparison
submission_avg = pd.DataFrame({
    'ProfileID': test_ids, 
    'RiskFlag': ensemble_preds_avg
})
submission_avg.to_csv('nn_submission_simple_avg.csv', index=False)

# Save weighted with default 0.5 threshold
submission_weighted = pd.DataFrame({
    'ProfileID': test_ids, 
    'RiskFlag': ensemble_preds_weighted
})
submission_weighted.to_csv('nn_submission_weighted.csv', index=False)

print("\n" + "="*80)
print("SUBMISSIONS SAVED")
print("="*80)
print(f"Main submission: nn_leaderboard_submission.csv (weighted + optimized threshold)")
print(f"Alternative 1: nn_submission_simple_avg.csv (simple average)")
print(f"Alternative 2: nn_submission_weighted.csv (weighted, threshold=0.5)")

print("\n" + "="*80)
print("PREDICTION STATISTICS")
print("="*80)
print(f"Optimized predictions - RiskFlag=1: {ensemble_preds_optimized.sum()} ({ensemble_preds_optimized.mean()*100:.2f}%)")
print(f"Simple avg predictions - RiskFlag=1: {ensemble_preds_avg.sum()} ({ensemble_preds_avg.mean()*100:.2f}%)")
print(f"Training set - RiskFlag=1: {y_train.sum()} ({y_train.mean()*100:.2f}%)")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("1. Submit 'nn_leaderboard_submission.csv' first (optimized approach)")
print("2. If plateau persists, try the alternatives")
print("3. Consider stacking with XGBoost/LightGBM/CatBoost for further improvement")
print("4. Feature selection might help if overfitting")
print("="*80)