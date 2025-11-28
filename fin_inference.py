import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from fin_preprocess_split import load_train_split

# We recompute the same split (same SEED) to get X_test, y_test
X_train, X_test, y_train, y_test, id_train, id_test = load_train_split()
print("Loaded preprocessed split. Test shape:", X_test.shape)

# Ask which model to demonstrate (NN, LOGREG, SVM)
model_name = input("Enter model to run on financial dataset (nn / logreg / svm): ").strip().lower()

if model_name == "nn":
    model = joblib.load("fin_nn_model.pkl")
    print("\nLoaded fin_nn_model.pkl")
elif model_name == "logreg":
    model = joblib.load("fin_logreg_model.pkl")
    print("\nLoaded fin_logreg_model.pkl")
elif model_name == "svm":
    model = joblib.load("fin_svm_model.pkl")
    print("\nLoaded fin_svm_model.pkl")
else:
    raise ValueError("Unknown model name. Use nn / logreg / svm.")

# Predict on test split
y_proba = None
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Compute metrics
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# ROC-AUC only if we have probabilities
if y_proba is not None:
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC-AUC: {auc:.4f}")
else:
    print("Model has no predict_proba; ROC-AUC not shown.")
