import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from fin_preprocess_split import load_train_split
from config import SEED

# Load preprocessed train/test (but we only use train part here)
X_train, X_test, y_train, y_test, id_train, id_test = load_train_split()

print("Training data shape:", X_train.shape)

# 1) Neural Network
nn = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.002,
    max_iter=400,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=15,
    learning_rate_init=0.003,
    learning_rate='adaptive',
    random_state=SEED,
    verbose=False
)
nn.fit(X_train, y_train)
joblib.dump(nn, "fin_nn_model.pkl")
print("Saved fin_nn_model.pkl")

# 2) Logistic Regression
logreg = LogisticRegression(
    C=1.5,
    max_iter=500,
    solver='liblinear'
)
logreg.fit(X_train, y_train)
joblib.dump(logreg, "fin_logreg_model.pkl")
print("Saved fin_logreg_model.pkl")

# 3) Linear SVM with probability via calibration
base_svm = LinearSVC(C=1.0, max_iter=1000, random_state=SEED)
svm = CalibratedClassifierCV(base_svm, cv=3)
svm.fit(X_train, y_train)
joblib.dump(svm, "fin_svm_model.pkl")
print("Saved fin_svm_model.pkl")

print("All three models trained and saved.")
