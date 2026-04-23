from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from pathlib import Path
#SCORE = 0.407
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
model = RandomForestRegressor()
x_train = pd.read_csv(DATA_DIR / "X_train_preprocessed.csv", sep=",")
x_test  = pd.read_csv(DATA_DIR / "X_test_preprocessed.csv", sep=",")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", sep=";")
if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
    y_train = y_train.iloc[:, 0]
model = RandomForestRegressor(n_estimators=500,random_state=42)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

# Fill Y with model predictions
sample_sub["Y"] = predictions

# Keep exact required columns/order
submission = sample_sub[["row_id", "Y"]]

submission.to_csv(DATA_DIR / "RFR_submission.csv", index=False)
print("Saved:", DATA_DIR / "RFR_submission.csv")