import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import TRAIN_PATH, TARGET, ID_COL

print("="*80)
print("EXPLORATORY DATA ANALYSIS - FINANCIAL RISK PROFILING")
print("="*80)

# Load data
train = pd.read_csv(TRAIN_PATH)

print(f"\n[1] Dataset Shape: {train.shape}")
print(f"\n[2] Columns:\n{train.columns.tolist()}")
print(f"\n[3] Data Types:\n{train.dtypes}")
print(f"\n[4] Missing Values:\n{train.isnull().sum()}")
print(f"\n[5] Target Distribution:\n{train[TARGET].value_counts()}")
print(f"\n[6] Class Balance: {train[TARGET].value_counts(normalize=True)}")

# Statistical summary
print(f"\n[7] Statistical Summary:\n{train.describe()}")

# Correlation with target
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if TARGET in numeric_cols:
    corr = train[numeric_cols].corr()[TARGET].sort_values(ascending=False)
    print(f"\n[8] Feature Correlation with {TARGET}:\n{corr}")

# Visualizations (optional, save to file)
plt.figure(figsize=(10, 6))
sns.countplot(data=train, x=TARGET)
plt.title('Target Distribution')
plt.savefig('target_distribution.png')
print("\n[9] Target distribution plot saved as 'target_distribution.png'")

print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)
