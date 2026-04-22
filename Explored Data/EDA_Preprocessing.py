"""
=============================================================
  Project 2 – Item Price Prediction
  EDA & Preprocessing code
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

# Rename anonymised columns to meaningful names
col_map = {
    "X1":  "Item_Identifier",
    "X2":  "Item_Weight",
    "X3":  "Item_Fat_Content",
    "X4":  "Item_Visibility",
    "X5":  "Item_Type",
    "X6":  "Item_MRP",
    "X7":  "Outlet_Identifier",
    "X8":  "Outlet_Establishment_Year",
    "X9":  "Outlet_Size",
    "X10": "Outlet_Location_Tier",
    "X11": "Outlet_Type",
    "Y":   "Item_Outlet_Price",
}
train.rename(columns=col_map, inplace=True)
test.rename(columns={k: v for k, v in col_map.items() if k != "Y"}, inplace=True)

# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
SEP = "=" * 60


print(SEP)
print("SHAPE & PREVIEW")
print(SEP)
print(f"Train shape : {train.shape}  ({train.shape[0]:,} rows, {train.shape[1]} columns)")
print(f"Test  shape : {test.shape}  ({test.shape[0]:,} rows, {test.shape[1]} columns)")
print("\n── Train head ──")
print(train.head())
print("\n── Test head ──")
print(test.head())


print(f"\n{SEP}")
print("DATA TYPES")
print(SEP)
print(train.dtypes)

print(f"\n{SEP}")
print("MISSING VALUES")
print(SEP)

def missing_report(df, name):
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print(f"  [{name}] No missing values found.")
    else:
        miss_pct = (miss / len(df) * 100).round(2)
        report = pd.DataFrame({"Missing Count": miss, "Missing %": miss_pct})
        print(f"  [{name}]")
        print(report.to_string())

missing_report(train, "Train")
print()
missing_report(test,  "Test")

print(f"\n{SEP}")
print("DUPLICATE ROWS")
print(SEP)
print(f"  Train duplicates : {train.duplicated().sum()}")
print(f"  Test  duplicates : {test.duplicated().sum()}")


print(f"\n{SEP}")
print("STATISTICAL SUMMARY (numeric columns)")
print(SEP)
print(train.describe().T.to_string())


print(f"\n{SEP}")
print("TARGET VARIABLE: Item_Outlet_Price")
print(SEP)
tgt = train["Item_Outlet_Price"]
print(f"  Min    : {tgt.min():.4f}")
print(f"  Max    : {tgt.max():.4f}")
print(f"  Mean   : {tgt.mean():.4f}")
print(f"  Median : {tgt.median():.4f}")
print(f"  Std    : {tgt.std():.4f}")
print(f"  Skew   : {tgt.skew():.4f}  {'(right-skewed)' if tgt.skew() > 0.5 else '(left-skewed)' if tgt.skew() < -0.5 else '(approx. normal)'}")
print(f"  Kurtosis: {tgt.kurtosis():.4f}")


print(f"\n{SEP}")
print("CATEGORICAL COLUMNS")
print(SEP)
cat_cols = train.select_dtypes(include=["object", "string"]).columns.tolist()
print(f"  Categorical columns: {cat_cols}\n")
for col in cat_cols:
    vals = sorted(train[col].dropna().unique())
    print(f"  {col} ({train[col].nunique()} unique):")
    print(f"    {vals}")
    vc = train[col].value_counts(dropna=False)
    print(f"    Value counts:\n{vc.to_string()}\n")


print(f"\n{SEP}")
print("NUMERIC COLUMNS: SKEW & OUTLIER FLAGS")
print(SEP)
num_cols = train.select_dtypes(include=np.number).columns.drop("Item_Outlet_Price").tolist()
for col in num_cols:
    skew  = train[col].skew()
    zeros = (train[col] == 0).sum()
    q1, q3 = train[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((train[col] < q1 - 1.5*iqr) | (train[col] > q3 + 1.5*iqr)).sum()
    print(f"  {col}:")
    print(f"    Skew={skew:.3f}  |  Zeros={zeros}  |  IQR Outliers={outliers}")


print(f"\n{SEP}")
print("NUMERIC CORRELATION WITH TARGET")
print(SEP)
corr = train[num_cols + ["Item_Outlet_Price"]].corr()["Item_Outlet_Price"].drop("Item_Outlet_Price").sort_values(ascending=False)
print(corr.to_string())


print(f"\n{SEP}")
print("CATEGORICAL FEATURE MEAN PRICE")
print(SEP)
cat_analysis_cols = ["Item_Fat_Content", "Item_Type", "Outlet_Size",
                     "Outlet_Location_Tier", "Outlet_Type", "Outlet_Identifier"]
for col in cat_analysis_cols:
    agg = (train.groupby(col)["Item_Outlet_Price"]
           .agg(["mean", "median", "count"])
           .round(3)
           .sort_values("mean", ascending=False))
    print(f"\n  {col} vs Item_Outlet_Price:")
    print(agg.to_string())

# ─────────────────────────────────────────────
# 3. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("  PREPROCESSING PIPELINE")
print(SEP)

def preprocess(df, is_train=True):
    df = df.copy()

    #drop non-predictive ID columns
    df.drop(columns=["Item_Identifier", "Outlet_Identifier"], inplace=True)

    #feature Engineering
    df["Outlet_Age"] = 2025 - df["Outlet_Establishment_Year"]
    df.drop(columns=["Outlet_Establishment_Year"], inplace=True)

    df["MRP_Bin"] = pd.cut(df["Item_MRP"], bins=4, labels=["Low", "Medium", "High", "Premium"])

    
    fat_map = {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(fat_map)
    print("  ✔ Item_Fat_Content standardised →", sorted(df["Item_Fat_Content"].unique()))

   
    weight_median = df["Item_Weight"].median()
    df["Item_Weight"].fillna(weight_median, inplace=True)
    print(f"  ✔ Item_Weight: filled {df['Item_Weight'].isnull().sum()} missing → median = {weight_median:.3f}")

   
    df["Outlet_Size"] = df.groupby("Outlet_Type")["Outlet_Size"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Medium")
    )
    print("  ✔ Outlet_Size: filled missing → mode per Outlet_Type")

   
    zero_vis = (df["Item_Visibility"] == 0).sum()
    vis_mean = df[df["Item_Visibility"] > 0]["Item_Visibility"].mean()
    df["Item_Visibility"] = df["Item_Visibility"].replace(0, vis_mean)
    print(f"  ✔ Item_Visibility: {zero_vis} zero(s) replaced with mean = {vis_mean:.5f}")

    #Encoding Categorical Variables
    le = LabelEncoder()
    df["Item_Fat_Content"] = le.fit_transform(df["Item_Fat_Content"])
    print("Label Encoded  : Item_Fat_Content")

    df["Outlet_Size"]          = df["Outlet_Size"].map({"Small": 0, "Medium": 1, "High": 2})
    df["Outlet_Location_Tier"] = df["Outlet_Location_Tier"].map({"Tier 1": 1, "Tier 2": 2, "Tier 3": 3})
    print("Ordinal Encoded: Outlet_Size, Outlet_Location_Tier")

    df = pd.get_dummies(df, columns=["Item_Type", "Outlet_Type", "MRP_Bin"], drop_first=False)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    print("One-Hot Encoded: Item_Type, Outlet_Type, MRP_Bin")

    return df


#run preprocessing on train and test
train_clean = preprocess(train, is_train=True)
test_clean  = preprocess(test,  is_train=False)

X_train = train_clean.drop(columns=["Item_Outlet_Price"])
y_train = train_clean["Item_Outlet_Price"]
test_clean = test_clean.reindex(columns=X_train.columns, fill_value=0)

print(f"\n  Train features : {X_train.shape}")
print(f"  Test  features : {test_clean.shape}")
print(f"  Target         : {y_train.shape}")

#Feature Scaling
scaler    = StandardScaler()
num_feats = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Age"]

X_train_scaled = X_train.copy()
test_scaled    = test_clean.copy()

X_train_scaled[num_feats] = scaler.fit_transform(X_train[num_feats])
test_scaled[num_feats]    = scaler.transform(test_clean[num_feats])
print(f"\n StandardScaler applied to: {num_feats}")
print(f" Missing values remaining : {X_train_scaled.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 4. SAVE PREPROCESSED DATA
# ─────────────────────────────────────────────
OUTPUT_DIR = BASE_DIR / "Data"
OUTPUT_DIR.mkdir(exist_ok=True)

X_train_scaled.to_csv(OUTPUT_DIR / "X_train_preprocessed.csv", index=False)
y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
test_scaled.to_csv(OUTPUT_DIR / "X_test_preprocessed.csv", index=False)

print(f"\n✔ Files saved to: {OUTPUT_DIR}")
print("   → X_train_preprocessed.csv")
print("   → y_train.csv")
print("   → X_test_preprocessed.csv")

