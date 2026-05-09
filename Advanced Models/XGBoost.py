import warnings
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#score = 0.383
warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
x_train = pd.read_csv(DATA_DIR / "X_train_preprocessed.csv", sep=",")
x_test  = pd.read_csv(DATA_DIR / "X_test_preprocessed.csv", sep=",")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", sep=";")
if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
    y_train = y_train.iloc[:, 0]


from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

param_dist = {
    "n_estimators": [600, 800, 1000],
    "learning_rate": [0.01, 0.02, 0.05],
    "max_depth": [3, 4, 5],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "gamma": [0, 0.1, 0.2],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1, 1.5, 2]
}

search = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    scoring="neg_mean_absolute_error",
    cv=3,
    n_jobs=-1,
    random_state=42
)

search.fit(x_train, y_train)

model = XGBRegressor(**search.best_params_, random_state=42)

model.fit(x_train, y_train)
predictions = model.predict(x_test)
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
# Fill Y with model predictions
sample_sub["Y"] = predictions

# Keep exact required columns/order
submission = sample_sub[["row_id", "Y"]]

submission.to_csv(DATA_DIR / "XGB_submission.csv", index=False)
print("Saved:", DATA_DIR / "XGB_submission.csv")