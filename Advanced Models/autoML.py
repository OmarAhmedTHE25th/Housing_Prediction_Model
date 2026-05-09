from pycaret.regression import *
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

x_train = pd.read_csv(DATA_DIR / "X_train_preprocessed.csv", sep=",")
x_test  = pd.read_csv(DATA_DIR / "X_test_preprocessed.csv", sep=",")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", sep=";")

if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
    y_train = y_train.iloc[:, 0]

data = pd.concat([x_train, y_train], axis=1)
# Rename all columns explicitly
data.columns = [f"feature_{i}" for i in range(data.shape[1] - 1)] + ["target"]
print(data.columns.tolist())  # verify "target" is last


setup(data=data, target="target")

best_model = compare_models()  # this already returns the best model directly
x_test.columns = [f"feature_{i}" for i in range(x_test.shape[1])]
predictions = predict_model(best_model, data=x_test)
save_model(best_model, str(DATA_DIR / "best_model"))

print("Best model:", best_model)