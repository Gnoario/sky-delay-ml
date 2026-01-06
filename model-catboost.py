import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, RocCurveDisplay
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"

AIRLINES_CSV = DATA_DIR / "airlines.csv"
AIRPORTS_CSV = DATA_DIR / "airports.csv"
FLIGHTS_CSV  = DATA_DIR / "flights.csv"

OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SAMPLE_N = 1_500_000 
RANDOM_STATE = 42
MISSING_TOKEN = "MISSING"

def to_hour_min_from_hhmm(val):
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

def sanitize_cat_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype("object")
        df[c] = df[c].where(df[c].notna(), MISSING_TOKEN)
        df[c] = df[c].astype(str)
        df.loc[df[c].isin(["nan", "NaN", "<NA>", "None"]), c] = MISSING_TOKEN
    return df

def stratified_sample(X, y, n, random_state=42):
    """Estratificado por y para manter a proporção de atrasos."""
    if n is None or len(X) <= n:
        return X, y
    tmp = X.copy()
    tmp["__y__"] = y.values
    counts = tmp["__y__"].value_counts().to_dict()
    total = len(tmp)
    n_pos = int(n * (counts.get(1, 0) / total))
    n_neg = n - n_pos
    pos = tmp[tmp["__y__"] == 1].sample(n=n_pos, random_state=random_state)
    neg = tmp[tmp["__y__"] == 0].sample(n=n_neg, random_state=random_state)
    sampled = pd.concat([pos, neg], axis=0).sample(frac=1, random_state=random_state)
    y_s = sampled["__y__"].astype(int)
    X_s = sampled.drop(columns="__y__")
    return X_s, y_s

airlines = pd.read_csv(AIRLINES_CSV) 
airports = pd.read_csv(AIRPORTS_CSV) 

print("Loaded:")
print("airlines:", airlines.shape)
print("airports:", airports.shape)

flights_usecols = [
    "YEAR","MONTH","DAY","DAY_OF_WEEK",
    "AIRLINE","FLIGHT_NUMBER",
    "ORIGIN_AIRPORT","DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE","SCHEDULED_ARRIVAL",
    "SCHEDULED_TIME","DISTANCE",
    "ARRIVAL_DELAY",
    "CANCELLED","DIVERTED",
]

flights = pd.read_csv(FLIGHTS_CSV, usecols=flights_usecols)
print("flights :", flights.shape)

for c in ["YEAR","MONTH","DAY","DAY_OF_WEEK","CANCELLED","DIVERTED"]:
    if c in flights.columns:
        flights[c] = pd.to_numeric(flights[c], errors="coerce").astype("Int16")

for c in ["SCHEDULED_DEPARTURE","SCHEDULED_ARRIVAL","SCHEDULED_TIME","DISTANCE","FLIGHT_NUMBER","ARRIVAL_DELAY"]:
    if c in flights.columns:
        flights[c] = pd.to_numeric(flights[c], errors="coerce").astype("float32")

airlines_ref = airlines.rename(columns={"IATA_CODE":"AIRLINE","AIRLINE":"AIRLINE_NAME"})[["AIRLINE","AIRLINE_NAME"]]
flights = flights.merge(airlines_ref, on="AIRLINE", how="left")

airports_origin = airports.rename(columns={
    "IATA_CODE":"ORIGIN_AIRPORT",
    "STATE":"ORIGIN_STATE",
})[["ORIGIN_AIRPORT","ORIGIN_STATE"]]
flights = flights.merge(airports_origin, on="ORIGIN_AIRPORT", how="left")

airports_dest = airports.rename(columns={
    "IATA_CODE":"DESTINATION_AIRPORT",
    "STATE":"DEST_STATE",
})[["DESTINATION_AIRPORT","DEST_STATE"]]
flights = flights.merge(airports_dest, on="DESTINATION_AIRPORT", how="left")

print("After merges:", flights.shape)

flights_model = flights[(flights["CANCELLED"] == 0) & (flights["DIVERTED"] == 0)].copy()
flights_model = flights_model[~pd.isna(flights_model["ARRIVAL_DELAY"])].copy()
flights_model["delayed"] = (flights_model["ARRIVAL_DELAY"] >= 15).astype(int)

print("Modeling rows:", flights_model.shape)
print("Delayed rate:", float(flights_model["delayed"].mean()).__round__(4))

flights_model = add_time_features(flights_model, "SCHEDULED_DEPARTURE", "sched_dep")
flights_model = add_time_features(flights_model, "SCHEDULED_ARRIVAL", "sched_arr")

flights_model["is_weekend"] = flights_model["DAY_OF_WEEK"].isin([6,7]).astype("int8")
flights_model["distance_bucket"] = flights_model["DISTANCE"].apply(distance_bucket)

feature_cols = [
    "YEAR","MONTH","DAY","DAY_OF_WEEK",
    "AIRLINE","AIRLINE_NAME",
    "FLIGHT_NUMBER",
    "ORIGIN_AIRPORT","ORIGIN_STATE",
    "DESTINATION_AIRPORT","DEST_STATE",
    "DISTANCE","SCHEDULED_TIME",
    "SCHEDULED_DEPARTURE","SCHEDULED_ARRIVAL",
    "sched_dep_hour","sched_dep_minute","sched_dep_period",
    "sched_arr_hour","sched_arr_minute","sched_arr_period",
    "is_weekend",
    "distance_bucket",
]
feature_cols = [c for c in feature_cols if c in flights_model.columns]

X = flights_model[feature_cols].copy()
y = flights_model["delayed"].copy()

print("X shape:", X.shape, "| y shape:", y.shape)

cat_features = [c for c in [
    "AIRLINE","AIRLINE_NAME",
    "ORIGIN_AIRPORT","ORIGIN_STATE",
    "DESTINATION_AIRPORT","DEST_STATE",
    "sched_dep_period","sched_arr_period",
    "distance_bucket",
] if c in X.columns]

X = sanitize_cat_cols(X, cat_features)

for c in X.columns:
    if c not in cat_features:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train delayed rate:", float(y_train.mean()).__round__(4))
print("Test  delayed rate:", float(y_test.mean()).__round__(4))

cat_features = [
    c for c in [
        "AIRLINE", "AIRLINE_NAME",
        "ORIGIN_AIRPORT", "ORIGIN_STATE",
        "DESTINATION_AIRPORT", "DEST_STATE",
        "sched_dep_period", "sched_arr_period",
        "distance_bucket",
    ] if c in X_train.columns
]

MISSING_TOKEN = "MISSING"

def sanitize_catboost_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype("object")
        df[c] = df[c].where(df[c].notna(), MISSING_TOKEN)
        df[c] = df[c].astype(str)
        df.loc[df[c].isin(["nan", "NaN", "<NA>", "None"]), c] = MISSING_TOKEN
    return df

X_train = sanitize_catboost_categoricals(X_train, cat_features)
X_test  = sanitize_catboost_categoricals(X_test, cat_features)

X_train, y_train = stratified_sample(X_train, y_train, TRAIN_SAMPLE_N, random_state=RANDOM_STATE)
print("✅ Train size after sampling:", X_train.shape, "| delayed rate:", float(y_train.mean()).__round__(4))

cat_idx = [X_train.columns.get_loc(c) for c in cat_features]

train_pool = Pool(X_train, y_train, cat_features=cat_idx)
test_pool  = Pool(X_test, y_test, cat_features=cat_idx)

neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
scale_pos_weight = neg / max(pos, 1)
print("Class balance:", {"neg": neg, "pos": pos, "scale_pos_weight": round(scale_pos_weight, 4)})

cb = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.06,
    depth=6,                     
    min_data_in_leaf=200,     
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=100,
    scale_pos_weight=scale_pos_weight,

    max_ctr_complexity=2,
    one_hot_max_size=16,
    border_count=128,

    bootstrap_type="Bernoulli",
    subsample=0.7,
    rsm=0.8,

    od_type="Iter",
    od_wait=50,

    allow_writing_files=False,

    # GPU optional settings
    task_type="GPU",
    devices="0",
)

cb.fit(train_pool, eval_set=test_pool, use_best_model=True)

proba = cb.predict_proba(test_pool)[:, 1]
pred = (proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, proba)
f1 = f1_score(y_test, pred)
rec = recall_score(y_test, pred)
prec = precision_score(y_test, pred, zero_division=0)

print("\n==============================")
print("Model: CatBoost (memory-safe)")
print(f"ROC-AUC  : {auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1       : {f1:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification report:\n", classification_report(y_test, pred, digits=4))

RocCurveDisplay.from_predictions(y_test, proba)
plt.title("ROC Curve - CatBoost (memory-safe)")
roc_path = OUTPUT_DIR / "roc_curve_catboost.png"
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n🖼️ Saved ROC curve to: {roc_path}")

thresholds = np.arange(0.05, 0.96, 0.01)
rows = []
for t in thresholds:
    p = (proba >= t).astype(int)
    rows.append([
        t,
        precision_score(y_test, p, zero_division=0),
        recall_score(y_test, p),
        f1_score(y_test, p),
    ])

thr_df = pd.DataFrame(rows, columns=["threshold","precision","recall","f1"])
best_row = thr_df.loc[thr_df["f1"].idxmax()]
best_thr = float(best_row["threshold"])

print("\n==============================")
print("🔧 Threshold optimization (maximize F1)")
print("==============================")
print("✅ Best threshold:", best_thr)
print(best_row)

plt.figure(figsize=(10,5))
plt.plot(thr_df["threshold"], thr_df["precision"], label="Precision")
plt.plot(thr_df["threshold"], thr_df["recall"], label="Recall")
plt.plot(thr_df["threshold"], thr_df["f1"], label="F1-score")
plt.axvline(best_thr, linestyle="--", label=f"Best threshold={best_thr:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision / Recall / F1 vs Threshold - CatBoost")
plt.legend()
plt.grid(True)
thr_path = OUTPUT_DIR / "threshold_curve_catboost.png"
plt.savefig(thr_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"🖼️ Saved threshold curve to: {thr_path}")

pred_opt = (proba >= best_thr).astype(int)
print("\n📊 Metrics @ optimized threshold")
print("Precision:", precision_score(y_test, pred_opt, zero_division=0))
print("Recall   :", recall_score(y_test, pred_opt))
print("F1-score :", f1_score(y_test, pred_opt))
print("Confusion matrix:\n", confusion_matrix(y_test, pred_opt))

model_path = OUTPUT_DIR / "catboost_model.cbm"
cb.save_model(str(model_path))
print(f"\n💾 Saved CatBoost model to: {model_path}")

thr_txt = OUTPUT_DIR / "catboost_best_threshold.txt"
thr_txt.write_text(str(best_thr))
print(f"💾 Saved best threshold to: {thr_txt}")
