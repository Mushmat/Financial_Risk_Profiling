import os

# Paths
BASE_DIR = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(BASE_DIR, "train_updated.csv")
TEST_PATH = os.path.join(BASE_DIR, "test_updated.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(BASE_DIR, "sample_submission_updated.csv")

# Target and ID columns
TARGET = 'RiskFlag'
ID_COL = 'ProfileID'

# Random seed
SEED = 42
 