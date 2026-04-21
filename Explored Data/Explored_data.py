import pandas as pd
import numpy as np
from pathlib import Path

# Get the project root based on this script's location
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

# Load the dataset
data_train = pd.read_csv(DATA_DIR / "train.csv")
data_test = pd.read_csv(DATA_DIR / "test.csv")

# Display the first few rows of the dataset
print("Training Data:")
print(data_train.head())
print("\n")
print("Test Data:")
print(data_test.head())
print("\n")
# Check for missing values
print("Missing values in training data:")
print(data_train.isnull().sum())
print("\n")
print("Missing values in test data:")
print(data_test.isnull().sum())
print("\n")
# Show missing values in their places
print("Training data with missing values:")
print(data_train[data_train.isnull().any(axis=1)])
print("\n")
print("Test data with missing values:")
print(data_test[data_test.isnull().any(axis=1)])
print("\n")
# Check the non numerical columns
print("Non-numerical columns in training data:")
print(data_train.select_dtypes(include=['object', 'string']).columns )
print("\n"+"Non-numerical columns in test data:")
print(data_test.select_dtypes(include=['object', 'string']).columns )

