# score: 0.476
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
#kernel is used to define the type of kernel to be used in the algorithm,
# rbf is used for non-linear data
# C is the penalty parameter C of the error term, it controls the trade-off between smooth decision boundary and fitting the training data
# gamma is the kernel coefficient for rbf kernel, it defines how far the influence of a single training example reaches,
# with low values meaning far and high values meaning close
# epsilon defines a "tube" around the predicted line.
# Points inside the tube don't contribute to the loss at all.
# Larger epsilon = wider tube = less sensitive to small errors.
x_train = pd.read_csv(DATA_DIR / "X_train_preprocessed.csv", sep=",")
x_test  = pd.read_csv(DATA_DIR / "X_test_preprocessed.csv", sep=",")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", sep=";")
if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
    y_train = y_train.iloc[:, 0]

X_train_part, X_val, y_train_part, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=42
)

model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
model.fit(X_train_part, y_train_part)

val_predictions = model.predict(X_val)
print("Local validation MAE:", mean_absolute_error(y_val, val_predictions))

model.fit(x_train, y_train)
predictions = model.predict(x_test)
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
# Fill Y with model predictions
sample_sub["Y"] = predictions

# Keep exact required columns/order
submission = sample_sub[["row_id", "Y"]]

submission.to_csv(DATA_DIR / "SVR_submission.csv", index=False)
print("Saved:", DATA_DIR / "SVR_submission.csv")