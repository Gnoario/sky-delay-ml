# ==========================================
# Tech Challenge MLET Fase 3 - V5.1
# CatBoost (GPU) - Temporal split + Rolling (no leakage) + Congestion features
# Train strategy: temporal balanced sampling per month (cap) to reduce overfitting
#
# Split:
#   Train months: [1..8]
#   Val   months: [9,10]   -> threshold calibration (F1/F2/F0.5 selectable)
#   Test  months: [11,12]  -> final report
#
# Features:
# - Route
# - Day-of-year + cyclical encodings (sin/cos)
# - HHMM -> hour/minute + periods
# - Congestion (scheduled volume) at origin/dest per day-hour (no delay use)
# - Rolling historical delay rates (shifted) per group:
#     airline, origin, dest, route, origin_hour, origin_dow
#   computed in a strictly temporal, leakage-free manner.
#
# Outputs:
# - models/catboost_model_v51.cbm
# - models/catboost_best_threshold_v51_<metric>.txt
# - models/roc_curve_catboost_v51.png
# - models/threshold_curve_catboost_v51_<metric>.png
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, RocCurveDisplay
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool

# --------------------------
# Config
# --------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"

AIRLINES_CSV = DATA_DIR / "airlines.csv"
AIRPORTS_CSV = DATA_DIR / "airports.csv"
FLIGHTS_CSV  = DATA_DIR / "flights.csv"

OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MISSING_TOKEN = "MISSING"

# Temporal split
TRAIN_MONTHS = list(range(1, 9))     # 1..8
VAL_MONTHS   = [9, 10]
TEST_MONTHS  = [11, 12]

# Sampling strategy (best for generalization)
USE_TRAIN_SAMPLING = True
MAX_SAMPLES_PER_MONTH = 200_000      # cap per month in train (keeps temporal balance)
# If you want slightly better metrics and can wait longer, try 250_000 or 300_000.

# Threshold metric: "f1", "f2", "f0.5"
THRESH_METRIC = "f1"  # change to "f2" or "f0.5"

# Toggles
USE_ROUTE = True
USE_AIRLINE_NAME = False
USE_FLIGHT_NUMBER = False            # usually hurts generalization
DROP_RAW_HHMM_COLS = True            # keep hour/minute/period; drop raw HHMM ints to avoid weird splits

USE_GPU = True
GPU_DEVICE = "0"

# CatBoost params (moderately regularized; should improve over V5 while avoiding overfit)
CB_PARAMS = dict(
    iterations=1400,
    learning_rate=0.06,
    depth=7,
    min_data_in_leaf=300,      # lower than your old 1500; better fit while still generalizing
    l2_leaf_reg=15,            # lower than 30; less underfit
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=100,
    od_type="Iter",
    od_wait=120,
    # CTR / categorical handling
    max_ctr_complexity=2,
    one_hot_max_size=16,
    border_count=128,
    # subsampling (GPU-safe options)
    bootstrap_type="Bernoulli",
    subsample=0.8,
    # IMPORTANT: rsm is not supported on GPU for Logloss (pairwise only) -> DO NOT SET rsm.
    allow_writing_files=False,
)

if USE_GPU:
    CB_PARAMS.update(dict(task_type="GPU", devices=GPU_DEVICE))

# --------------------------
# Helpers
# --------------------------
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

def make_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create flight_date and doy + cyclical encodings.
    Uses YEAR/MONTH/DAY. Invalid dates become NaT and then dropped.
    """
    df = df.copy()
    df["flight_date"] = pd.to_datetime(
        dict(year=df["YEAR"].astype("int32"), month=df["MONTH"].astype("int32"), day=df["DAY"].astype("int32")),
        errors="coerce"
    )
    df = df[~df["flight_date"].isna()].copy()
    df["day_of_year"] = df["flight_date"].dt.dayofyear.astype("int16")

    # cyclical encodings
    # using 365.25 is okay but keep simple 365 for this dataset
    two_pi = 2.0 * np.pi
    df["doy_sin"] = np.sin(two_pi * df["day_of_year"] / 365.0).astype("float32")
    df["doy_cos"] = np.cos(two_pi * df["day_of_year"] / 365.0).astype("float32")
    return df

def add_route(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ROUTE"] = (df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str))
    return df

def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Congestion proxies using only scheduled information:
    - origin_day_hour_volume: count of flights scheduled to depart from origin on that date/hour
    - dest_day_hour_volume:   count of flights scheduled to arrive at dest on that date/hour
    """
    df = df.copy()
    # Ensure hour columns exist
    if "sched_dep_hour" not in df.columns:
        raise ValueError("sched_dep_hour missing; call add_time_features first")
    if "sched_arr_hour" not in df.columns:
        raise ValueError("sched_arr_hour missing; call add_time_features first")

    # origin volume by date+hour
    g1 = df.groupby(["flight_date", "ORIGIN_AIRPORT", "sched_dep_hour"], dropna=False).size()
    df["origin_day_hour_volume"] = df.set_index(["flight_date", "ORIGIN_AIRPORT", "sched_dep_hour"]).index.map(g1).astype("float32")

    # dest volume by date+hour
    g2 = df.groupby(["flight_date", "DESTINATION_AIRPORT", "sched_arr_hour"], dropna=False).size()
    df["dest_day_hour_volume"] = df.set_index(["flight_date", "DESTINATION_AIRPORT", "sched_arr_hour"]).index.map(g2).astype("float32")

    # Optional: log scale (often helps trees)
    df["origin_day_hour_logvol"] = np.log1p(df["origin_day_hour_volume"]).astype("float32")
    df["dest_day_hour_logvol"] = np.log1p(df["dest_day_hour_volume"]).astype("float32")

    return df

def _rolling_mean_shifted(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """
    Strict no-leakage rolling mean:
    rolling mean over previous values only (shift 1).

    min_periods must be <= window.
    """
    if min_periods is None:
        # sensible default: require at least 3 points, but never more than window
        min_periods = max(3, min(20, window))
    else:
        min_periods = min(min_periods, window)

    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def add_rolling_target_rate(df: pd.DataFrame, group_cols: list[str], out_col: str, windows=(7, 30)) -> pd.DataFrame:
    """
    Adds rolling historical delay rate features per group, leakage-free, time-ordered.
    Creates columns:
      out_col_w7, out_col_w30 (by default)
    """
    df = df.copy()
    df = df.sort_values("flight_date").copy()

    for w in windows:
        col = f"{out_col}_w{w}"
        df[col] = (
            df.groupby(group_cols, dropna=False)["delayed"]
              .apply(lambda s: _rolling_mean_shifted(s, window=w))  # now safe
              .reset_index(level=group_cols, drop=True)
              .astype("float32")
        )

    return df


def beta_fscore(y_true, y_pred, beta: float) -> float:
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    if prec == 0 and rec == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * (prec * rec) / (b2 * prec + rec + 1e-12)

def optimize_threshold(y_true, proba, metric="f1"):
    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred)
        f1v = f1_score(y_true, pred)

        if metric == "f2":
            score = beta_fscore(y_true, pred, beta=2.0)
        elif metric == "f0.5":
            score = beta_fscore(y_true, pred, beta=0.5)
        else:
            score = f1v

        rows.append([t, prec, rec, f1v, score])

    df = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1", "score"])
    best = df.loc[df["score"].idxmax()].copy()
    best_thr = float(best["threshold"])
    return best_thr, df, best

def temporal_balanced_sample_per_month(X: pd.DataFrame, y: pd.Series, month_col: str, max_per_month: int, random_state=42):
    """
    Sample up to max_per_month rows for each month, preserving that month's class ratio (no base-rate distortion).
    """
    tmp = X.copy()
    tmp["__y__"] = y.values

    if month_col not in tmp.columns:
        raise ValueError(f"month_col '{month_col}' not found in X.")

    parts = []
    for m, g in tmp.groupby(month_col, dropna=False):
        if len(g) <= max_per_month:
            parts.append(g)
            continue

        # preserve monthly class ratio
        y_counts = g["__y__"].value_counts().to_dict()
        total = len(g)
        n_pos = int(max_per_month * (y_counts.get(1, 0) / total))
        n_neg = max_per_month - n_pos

        pos = g[g["__y__"] == 1].sample(n=min(n_pos, y_counts.get(1, 0)), random_state=random_state)
        neg = g[g["__y__"] == 0].sample(n=min(n_neg, y_counts.get(0, 0)), random_state=random_state)
        sampled = pd.concat([pos, neg], axis=0)

        # If due to min constraints we got fewer rows, top up randomly from remaining rows (still within month)
        if len(sampled) < max_per_month:
            remaining = g.drop(sampled.index)
            need = max_per_month - len(sampled)
            if need > 0 and len(remaining) > 0:
                sampled = pd.concat([sampled, remaining.sample(n=min(need, len(remaining)), random_state=random_state)], axis=0)

        parts.append(sampled)

    out = pd.concat(parts, axis=0).sample(frac=1, random_state=random_state)
    y_s = out["__y__"].astype(int)
    X_s = out.drop(columns="__y__")
    return X_s, y_s

# --------------------------
# Load data (memory-friendly)
# --------------------------
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

# Cast to compact types
for c in ["YEAR","MONTH","DAY","DAY_OF_WEEK","CANCELLED","DIVERTED"]:
    flights[c] = pd.to_numeric(flights[c], errors="coerce").astype("Int16")

for c in ["SCHEDULED_DEPARTURE","SCHEDULED_ARRIVAL","SCHEDULED_TIME","DISTANCE","FLIGHT_NUMBER","ARRIVAL_DELAY"]:
    flights[c] = pd.to_numeric(flights[c], errors="coerce").astype("float32")

# Reference joins (lightweight)
airlines_ref = airlines.rename(columns={"IATA_CODE":"AIRLINE","AIRLINE":"AIRLINE_NAME"})[["AIRLINE","AIRLINE_NAME"]]
flights = flights.merge(airlines_ref, on="AIRLINE", how="left")

airports_origin = airports.rename(columns={"IATA_CODE":"ORIGIN_AIRPORT","STATE":"ORIGIN_STATE"})[["ORIGIN_AIRPORT","ORIGIN_STATE"]]
flights = flights.merge(airports_origin, on="ORIGIN_AIRPORT", how="left")

airports_dest = airports.rename(columns={"IATA_CODE":"DESTINATION_AIRPORT","STATE":"DEST_STATE"})[["DESTINATION_AIRPORT","DEST_STATE"]]
flights = flights.merge(airports_dest, on="DESTINATION_AIRPORT", how="left")

print("After merges:", flights.shape)

# --------------------------
# Filter + target
# --------------------------
flights_model = flights[(flights["CANCELLED"] == 0) & (flights["DIVERTED"] == 0)].copy()
flights_model = flights_model[~pd.isna(flights_model["ARRIVAL_DELAY"])].copy()
flights_model["delayed"] = (flights_model["ARRIVAL_DELAY"] >= 15).astype(int)

print("Modeling rows:", flights_model.shape)
print("Delayed rate:", float(flights_model["delayed"].mean()).__round__(4))

# --------------------------
# Feature engineering
# --------------------------
flights_model = make_date_cols(flights_model)

flights_model = add_time_features(flights_model, "SCHEDULED_DEPARTURE", "sched_dep")
flights_model = add_time_features(flights_model, "SCHEDULED_ARRIVAL", "sched_arr")

flights_model["is_weekend"] = flights_model["DAY_OF_WEEK"].isin([6, 7]).astype("int8")
flights_model["distance_bucket"] = flights_model["DISTANCE"].apply(distance_bucket)

if USE_ROUTE:
    flights_model = add_route(flights_model)

# congestion features (scheduled volume)
flights_model = add_congestion_features(flights_model)

# rolling target rates (no leakage, time-ordered)
# You can tune windows; (7, 30) works well as short/medium memory.
flights_model = add_rolling_target_rate(flights_model, ["AIRLINE"], "te_airline", windows=(7, 30))
flights_model = add_rolling_target_rate(flights_model, ["ORIGIN_AIRPORT"], "te_origin", windows=(7, 30))
flights_model = add_rolling_target_rate(flights_model, ["DESTINATION_AIRPORT"], "te_dest", windows=(7, 30))
flights_model = add_rolling_target_rate(flights_model, ["ORIGIN_AIRPORT","DESTINATION_AIRPORT"], "te_route", windows=(7, 30))
flights_model = add_rolling_target_rate(flights_model, ["ORIGIN_AIRPORT","sched_dep_hour"], "te_origin_hour", windows=(7, 30))
flights_model = add_rolling_target_rate(flights_model, ["ORIGIN_AIRPORT","DAY_OF_WEEK"], "te_origin_dow", windows=(30,))

# --------------------------
# Select features
# --------------------------
base_features = [
    "YEAR","MONTH","DAY","DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT","ORIGIN_STATE",
    "DESTINATION_AIRPORT","DEST_STATE",
    "DISTANCE","SCHEDULED_TIME",
    "sched_dep_hour","sched_dep_minute","sched_dep_period",
    "sched_arr_hour","sched_arr_minute","sched_arr_period",
    "is_weekend",
    "distance_bucket",
    "day_of_year","doy_sin","doy_cos",
    # congestion
    "origin_day_hour_logvol","dest_day_hour_logvol",
    # rolling TE
    "te_airline_w7","te_airline_w30",
    "te_origin_w7","te_origin_w30",
    "te_dest_w7","te_dest_w30",
    "te_route_w7","te_route_w30",
    "te_origin_hour_w7","te_origin_hour_w30",
    "te_origin_dow_w30",
]

if USE_ROUTE:
    base_features.append("ROUTE")

if USE_AIRLINE_NAME and "AIRLINE_NAME" in flights_model.columns:
    base_features.append("AIRLINE_NAME")

if USE_FLIGHT_NUMBER and "FLIGHT_NUMBER" in flights_model.columns:
    base_features.append("FLIGHT_NUMBER")

# Optionally include raw HHMM fields (often harms generalization)
if not DROP_RAW_HHMM_COLS:
    base_features += ["SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL"]

# Keep only existing
feature_cols = [c for c in base_features if c in flights_model.columns]

X_all = flights_model[feature_cols].copy()
y_all = flights_model["delayed"].copy()

print("X shape:", X_all.shape, "| y shape:", y_all.shape)

# --------------------------
# Temporal split (strict)
# --------------------------
train_mask = flights_model["MONTH"].isin(TRAIN_MONTHS)
val_mask   = flights_model["MONTH"].isin(VAL_MONTHS)
test_mask  = flights_model["MONTH"].isin(TEST_MONTHS)

X_train, y_train = X_all[train_mask].copy(), y_all[train_mask].copy()
X_val,   y_val   = X_all[val_mask].copy(),   y_all[val_mask].copy()
X_test,  y_test  = X_all[test_mask].copy(),  y_all[test_mask].copy()

print(f"📅 Temporal split: train {TRAIN_MONTHS} | val {VAL_MONTHS} | test {TEST_MONTHS}")
print("Train delayed rate:", float(y_train.mean()).__round__(4))
print("Val   delayed rate:", float(y_val.mean()).__round__(4))
print("Test  delayed rate:", float(y_test.mean()).__round__(4))

# --------------------------
# Categorical setup
# --------------------------
cat_features = [
    "AIRLINE",
    "ORIGIN_AIRPORT","ORIGIN_STATE",
    "DESTINATION_AIRPORT","DEST_STATE",
    "sched_dep_period","sched_arr_period",
    "distance_bucket",
]
if USE_ROUTE:
    cat_features.append("ROUTE")
if USE_AIRLINE_NAME and "AIRLINE_NAME" in X_train.columns:
    cat_features.append("AIRLINE_NAME")
if USE_FLIGHT_NUMBER and "FLIGHT_NUMBER" in X_train.columns:
    # flight number as categorical is often too granular; but if enabled, keep consistent type
    cat_features.append("FLIGHT_NUMBER")

cat_features = [c for c in cat_features if c in X_train.columns]

# sanitize cats (CatBoost can't accept NaN for categorical)
X_train = sanitize_cat_cols(X_train, cat_features)
X_val   = sanitize_cat_cols(X_val, cat_features)
X_test  = sanitize_cat_cols(X_test, cat_features)

# numeric to float32
for df_ in (X_train, X_val, X_test):
    for c in df_.columns:
        if c not in cat_features:
            df_[c] = pd.to_numeric(df_[c], errors="coerce").astype("float32")

# Fill remaining NaNs in numerics with global prior from TRAIN only
train_prior = float(y_train.mean())
num_cols = [c for c in X_train.columns if c not in cat_features]
# rolling features might be NaN early; fill with train prior
for c in num_cols:
    X_train[c] = X_train[c].fillna(train_prior).astype("float32")
    X_val[c]   = X_val[c].fillna(train_prior).astype("float32")
    X_test[c]  = X_test[c].fillna(train_prior).astype("float32")

# --------------------------
# Train sampling (best approach)
# --------------------------
if USE_TRAIN_SAMPLING:
    # We sample within TRAIN only, by month, preserving each month's class ratio.
    # This reduces dominance of high-volume months and improves generalization.
    X_train_s, y_train_s = temporal_balanced_sample_per_month(
        X_train, y_train,
        month_col="MONTH",
        max_per_month=MAX_SAMPLES_PER_MONTH,
        random_state=RANDOM_STATE
    )
    # ensure types still good
    X_train_s = sanitize_cat_cols(X_train_s, cat_features)
    for c in X_train_s.columns:
        if c not in cat_features:
            X_train_s[c] = pd.to_numeric(X_train_s[c], errors="coerce").astype("float32").fillna(train_prior)
    X_train, y_train = X_train_s, y_train_s
    print(f"✅ Train size after temporal-balanced sampling: {X_train.shape} | delayed rate: {float(y_train.mean()).__round__(4)}")
else:
    print("ℹ️ Train sampling disabled; using full TRAIN set.")

cat_idx = [X_train.columns.get_loc(c) for c in cat_features]

train_pool = Pool(X_train, y_train, cat_features=cat_idx)
val_pool   = Pool(X_val, y_val, cat_features=cat_idx)
test_pool  = Pool(X_test, y_test, cat_features=cat_idx)

# Class weights (balanced) without distorting base-rate
# NOTE: CatBoost has auto_class_weights, but on GPU some setups behave differently.
# We'll set scale_pos_weight based on the sampled TRAIN distribution (still realistic).
neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
scale_pos_weight = neg / max(pos, 1)
print("Class balance:", {"neg": neg, "pos": pos, "scale_pos_weight": round(scale_pos_weight, 4)})

cb = CatBoostClassifier(
    **CB_PARAMS,
    scale_pos_weight=scale_pos_weight,
)

# --------------------------
# Fit (eval on VAL)
# --------------------------
cb.fit(train_pool, eval_set=val_pool, use_best_model=True)

# --------------------------
# Evaluate on TEST (default threshold 0.5)
# --------------------------
proba_test = cb.predict_proba(test_pool)[:, 1]
pred_test_05 = (proba_test >= 0.5).astype(int)

auc_test = roc_auc_score(y_test, proba_test)
f1_test = f1_score(y_test, pred_test_05)
rec_test = recall_score(y_test, pred_test_05)
prec_test = precision_score(y_test, pred_test_05, zero_division=0)

print("\n==============================")
print("Model: CatBoost (V5.1 rolling + temporal + congestion)")
print(f"Toggles: {{'USE_GPU': {USE_GPU}, 'USE_ROUTE': {USE_ROUTE}, 'USE_AIRLINE_NAME': {USE_AIRLINE_NAME}, 'USE_FLIGHT_NUMBER': {USE_FLIGHT_NUMBER}, 'THRESH_METRIC': '{THRESH_METRIC}'}}")
print(f"ROC-AUC  : {auc_test:.4f}")
print(f"Precision: {prec_test:.4f}")
print(f"Recall   : {rec_test:.4f}")
print(f"F1       : {f1_test:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, pred_test_05))
print("\nClassification report:\n", classification_report(y_test, pred_test_05, digits=4))

# ROC curve (TEST)
RocCurveDisplay.from_predictions(y_test, proba_test)
plt.title("ROC Curve - CatBoost V5.1 (TEST)")
roc_path = OUTPUT_DIR / "roc_curve_catboost_v51.png"
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n🖼️ Saved ROC curve to: {roc_path}")

# --------------------------
# Threshold optimization on VAL (then apply to TEST)
# --------------------------
proba_val = cb.predict_proba(val_pool)[:, 1]
best_thr, thr_df, best_row = optimize_threshold(y_val, proba_val, metric=THRESH_METRIC)

print("\n==============================")
print(f"🔧 Threshold optimization (maximize {THRESH_METRIC.upper()}) on VAL")
print("==============================")
print("✅ Best threshold:", best_thr)
print(best_row)

plt.figure(figsize=(10, 5))
plt.plot(thr_df["threshold"], thr_df["precision"], label="Precision")
plt.plot(thr_df["threshold"], thr_df["recall"], label="Recall")
plt.plot(thr_df["threshold"], thr_df["f1"], label="F1")
plt.plot(thr_df["threshold"], thr_df["score"], label=f"{THRESH_METRIC.upper()} (score)")
plt.axvline(best_thr, linestyle="--", label=f"Best={best_thr:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title(f"Precision/Recall/F1 vs Threshold (VAL) - Optimize {THRESH_METRIC.upper()}")
plt.legend()
plt.grid(True)
thr_path = OUTPUT_DIR / f"threshold_curve_catboost_v51_{THRESH_METRIC}.png"
plt.savefig(thr_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"🖼️ Saved threshold curve to: {thr_path}")

pred_test_opt = (proba_test >= best_thr).astype(int)

prec_opt = precision_score(y_test, pred_test_opt, zero_division=0)
rec_opt = recall_score(y_test, pred_test_opt)
f1_opt = f1_score(y_test, pred_test_opt)
f2_opt = beta_fscore(y_test, pred_test_opt, beta=2.0)
f05_opt = beta_fscore(y_test, pred_test_opt, beta=0.5)

print("\n📊 TEST metrics @ optimized threshold (chosen on VAL)")
print("Precision:", prec_opt)
print("Recall   :", rec_opt)
print("F1-score :", f1_opt)
print("F2      :", f2_opt)
print("F0.5    :", f05_opt)
print("Confusion matrix:\n", confusion_matrix(y_test, pred_test_opt))

# --------------------------
# Save model + threshold
# --------------------------
model_path = OUTPUT_DIR / "catboost_model_v51.cbm"
cb.save_model(str(model_path))
print(f"\n💾 Saved CatBoost model to: {model_path}")

thr_txt = OUTPUT_DIR / f"catboost_best_threshold_v51_{THRESH_METRIC}.txt"
thr_txt.write_text(str(best_thr))
print(f"💾 Saved best threshold to: {thr_txt}")
