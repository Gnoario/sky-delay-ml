
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    confusion_matrix, classification_report, RocCurveDisplay
)

import joblib
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"

AIRLINES_CSV = DATA_DIR / "airlines.csv"
AIRPORTS_CSV = DATA_DIR / "airports.csv"
FLIGHTS_CSV  = DATA_DIR / "flights.csv"

OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


airlines = pd.read_csv(AIRLINES_CSV)
airports = pd.read_csv(AIRPORTS_CSV)
flights  = pd.read_csv(FLIGHTS_CSV)

print("Loaded:")
print("airlines:", airlines.shape)
print("airports:", airports.shape)
print("flights :", flights.shape)

def to_hour_min_from_hhmm(val):
    """Convert HHMM integer/string -> (hour, minute). Returns (nan, nan) if invalid."""
    if pd.isna(val):
        return (np.nan, np.nan)
    try:
        s = str(int(val)).zfill(4)
        hh = int(s[:2])
        mm = int(s[2:])
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return (np.nan, np.nan)
        return (hh, mm)
    except Exception:
        return (np.nan, np.nan)

def add_time_features(df, col, prefix):
    hm = df[col].apply(to_hour_min_from_hhmm)
    df[f"{prefix}_hour"] = hm.apply(lambda x: x[0])
    df[f"{prefix}_minute"] = hm.apply(lambda x: x[1])

    def period(h):
        if pd.isna(h):
            return np.nan
        h = int(h)
        if 5 <= h <= 11:
            return "morning"
        if 12 <= h <= 17:
            return "afternoon"
        if 18 <= h <= 22:
            return "evening"
        return "night"
    df[f"{prefix}_period"] = df[f"{prefix}_hour"].apply(period)
    return df

def distance_bucket(d):
    if pd.isna(d):
        return np.nan
    if d < 500:
        return "short"
    if d < 1500:
        return "medium"
    return "long"


flights = flights.merge(
    airlines.rename(columns={"IATA_CODE": "AIRLINE", "AIRLINE": "AIRLINE_NAME"}),
    on="AIRLINE",
    how="left"
)

airports_origin = airports.rename(columns={
    "IATA_CODE": "ORIGIN_AIRPORT",
    "AIRPORT": "ORIGIN_AIRPORT_NAME",
    "CITY": "ORIGIN_CITY",
    "STATE": "ORIGIN_STATE",
    "COUNTRY": "ORIGIN_COUNTRY",
    "LATITUDE": "ORIGIN_LATITUDE",
    "LONGITUDE": "ORIGIN_LONGITUDE"
})
flights = flights.merge(airports_origin, on="ORIGIN_AIRPORT", how="left")

airports_dest = airports.rename(columns={
    "IATA_CODE": "DESTINATION_AIRPORT",
    "AIRPORT": "DEST_AIRPORT_NAME",
    "CITY": "DEST_CITY",
    "STATE": "DEST_STATE",
    "COUNTRY": "DEST_COUNTRY",
    "LATITUDE": "DEST_LATITUDE",
    "LONGITUDE": "DEST_LONGITUDE"
})
flights = flights.merge(airports_dest, on="DESTINATION_AIRPORT", how="left")

print("After merges:", flights.shape)

flights_model = flights[(flights["CANCELLED"] == 0) & (flights["DIVERTED"] == 0)].copy()

flights_model = flights_model[~flights_model["ARRIVAL_DELAY"].isna()].copy()
flights_model["delayed"] = (flights_model["ARRIVAL_DELAY"] >= 15).astype(int)

print("Modeling rows:", flights_model.shape)
print("Delayed rate:", flights_model["delayed"].mean().round(4))

flights_model = add_time_features(flights_model, "SCHEDULED_DEPARTURE", "sched_dep")
flights_model = add_time_features(flights_model, "SCHEDULED_ARRIVAL", "sched_arr")

flights_model["is_weekend"] = flights_model["DAY_OF_WEEK"].isin([6, 7]).astype(int)
flights_model["distance_bucket"] = flights_model["DISTANCE"].apply(distance_bucket)

LEAKAGE_COLS = [
    "ARRIVAL_DELAY", "DEPARTURE_DELAY",
    "DEPARTURE_TIME", "WHEELS_OFF", "WHEELS_ON", "ARRIVAL_TIME",
    "TAXI_OUT", "TAXI_IN", "ELAPSED_TIME", "AIR_TIME",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
    "CANCELLATION_REASON",
]
OPTIONAL_DROP = ["TAIL_NUMBER"]  # high-leak/ID-like

target_col = "delayed"

feature_cols = [
    "YEAR", "MONTH", "DAY", "DAY_OF_WEEK",
    "AIRLINE",
    "FLIGHT_NUMBER",
    "ORIGIN_AIRPORT", "ORIGIN_STATE",
    "DESTINATION_AIRPORT", "DEST_STATE",
    "DISTANCE", "SCHEDULED_TIME",
    "SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL",
    "sched_dep_hour", "sched_dep_minute", "sched_dep_period",
    "sched_arr_hour", "sched_arr_minute", "sched_arr_period",
    "is_weekend",
    "distance_bucket",
]
feature_cols = [c for c in feature_cols if c in flights_model.columns]

df = flights_model.drop(
    columns=[c for c in (LEAKAGE_COLS + OPTIONAL_DROP) if c in flights_model.columns],
    errors="ignore"
)

X = df[feature_cols].copy()
y = df[target_col].copy()

print("X shape:", X.shape, "| y shape:", y.shape)


if "FLIGHT_NUMBER" in X.columns:
    X["FLIGHT_NUMBER"] = pd.to_numeric(X["FLIGHT_NUMBER"], errors="coerce")

categorical_features = [
    c for c in [
        "AIRLINE", "AIRLINE_NAME",
        "ORIGIN_AIRPORT", "ORIGIN_STATE",
        "DESTINATION_AIRPORT", "DEST_STATE",
        "sched_dep_period", "sched_arr_period",
        "distance_bucket",
    ] if c in X.columns
]

for c in categorical_features:
    X[c] = X[c].astype("object")
    X[c] = X[c].where(~pd.isna(X[c]), np.nan)
    X.loc[X[c].notna(), c] = X.loc[X[c].notna(), c].astype(str)

X = X.replace({pd.NA: np.nan})

numeric_features = [c for c in X.columns if c not in categorical_features]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = X_train.replace({pd.NA: np.nan})
X_test  = X_test.replace({pd.NA: np.nan})

print("Train delayed rate:", y_train.mean().round(4))
print("Test  delayed rate:", y_test.mean().round(4))

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # sparse by default
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

log_reg = LogisticRegression(
    max_iter=200,
    class_weight="balanced",
    solver="saga",
)

sgd_log = SGDClassifier(
    loss="log_loss",
    alpha=1e-5,
    penalty="l2",
    class_weight="balanced",
    max_iter=30,
    tol=1e-3,
    random_state=42
)

# rf = RandomForestClassifier(
#     n_estimators=300,
#     random_state=42,
#     n_jobs=-1,
#     class_weight="balanced_subsample",
#     max_depth=None,
#     min_samples_leaf=1
# )

models = {
    "LogisticRegression": log_reg,
    "SGD_LogLoss": sgd_log,
}


def evaluate_model(name, clf, X_train, y_train, X_test, y_test):
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf)
    ])

    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    # Using 0.5 threshold or 0.35
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print("\n==============================")
    print(f"Model: {name}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1     : {f1:.4f}")
    print(f"Recall : {rec:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    return pipe, auc


best_name = None
best_pipe = None
best_auc = -1

for name, clf in models.items():
    pipe, auc = evaluate_model(name, clf, X_train, y_train, X_test, y_test)
    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_pipe = pipe

print("\n✅ Best model:", best_name, "| Best ROC-AUC:", round(best_auc, 4))

RocCurveDisplay.from_estimator(best_pipe, X_test, y_test)
plt.title(f"ROC Curve - Best model: {best_name}")
plt.show()

model_path = OUTPUT_DIR / "best_model.joblib"
joblib.dump(best_pipe, model_path)
print(f"\n💾 Saved best model to: {model_path}")
