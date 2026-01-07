import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool
import joblib

# ============================================================
# V6.2 - Calibrated + Prior-shift (base-rate drift) + Constraint
# Goal: choose threshold that maximizes recall subject to precision>=MIN_PRECISION_VAL
# using VAL probabilities adjusted to TEST prior (pi_test).
# ============================================================

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

# Sampling strategy
USE_TRAIN_SAMPLING = True
MAX_SAMPLES_PER_MONTH = 200_000  # cap per month in train (keeps temporal balance)

# Target constraint: precision >= this, maximize recall (chosen on VAL, adjusted to TEST prior)
MIN_PRECISION_VAL = 0.66

# Toggles
USE_ROUTE = True
USE_AIRLINE_NAME = False
USE_FLIGHT_NUMBER = False
DROP_RAW_HHMM_COLS = True

USE_GPU = True
GPU_DEVICE = "0"

# scale_pos_weight strategy
SPW_MULTIPLIER = 0.85  # your best run so far
# You can try 0.80..0.90 if you want a tiny grid later.

# CatBoost params (moderately regularized)
CB_PARAMS = dict(
    iterations=2000,
    learning_rate=0.06,
    depth=8,
    min_data_in_leaf=200,
    l2_leaf_reg=15,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=100,
    od_type="Iter",
    od_wait=200,
    max_ctr_complexity=2,
    one_hot_max_size=16,
    border_count=128,
    bootstrap_type="Bernoulli",
    subsample=0.7,
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

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def prior_shift_adjust(proba, pi_source, pi_target, eps=1e-6):
    """
    Adjust calibrated probabilities from source prior (pi_source) to target prior (pi_target)
    using log-odds shift. This helps with base-rate drift between VAL and TEST.
    """
    ls = np.log((pi_source + eps) / (1 - pi_source + eps))
    lt = np.log((pi_target + eps) / (1 - pi_target + eps))
    return sigmoid(logit(proba, eps=eps) + (lt - ls))

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
    df = df.copy()
    df["flight_date"] = pd.to_datetime(
        dict(
            year=df["YEAR"].astype("int32"),
            month=df["MONTH"].astype("int32"),
            day=df["DAY"].astype("int32"),
        ),
        errors="coerce",
    )
    df = df[~df["flight_date"].isna()].copy()

    df["day_of_year"] = df["flight_date"].dt.dayofyear.astype("int16")

    two_pi = 2.0 * np.pi
    df["doy_sin"] = np.sin(two_pi * df["day_of_year"] / 365.0).astype("float32")
    df["doy_cos"] = np.cos(two_pi * df["day_of_year"] / 365.0).astype("float32")
    return df

def add_route(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ROUTE"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)
    return df

def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sched_dep_hour" not in df.columns or "sched_arr_hour" not in df.columns:
        raise ValueError("Missing hour columns; call add_time_features first")

    g1 = df.groupby(["flight_date", "ORIGIN_AIRPORT", "sched_dep_hour"], dropna=False).size()
    df["origin_day_hour_volume"] = (
        df.set_index(["flight_date", "ORIGIN_AIRPORT", "sched_dep_hour"]).index.map(g1).astype("float32")
    )

    g2 = df.groupby(["flight_date", "DESTINATION_AIRPORT", "sched_arr_hour"], dropna=False).size()
    df["dest_day_hour_volume"] = (
        df.set_index(["flight_date", "DESTINATION_AIRPORT", "sched_arr_hour"]).index.map(g2).astype("float32")
    )

    df["origin_day_hour_logvol"] = np.log1p(df["origin_day_hour_volume"]).astype("float32")
    df["dest_day_hour_logvol"] = np.log1p(df["dest_day_hour_volume"]).astype("float32")
    return df

def _rolling_mean_shifted(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    # ensure min_periods <= window
    if min_periods is None:
        min_periods = max(3, min(window, 20))
    else:
        min_periods = min(min_periods, window)
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()

def add_rolling_target_rate(df: pd.DataFrame, group_cols: list[str], out_col: str, windows=(7, 30)) -> pd.DataFrame:
    df = df.copy().sort_values("flight_date")
    for w in windows:
        col = f"{out_col}_w{w}"
        df[col] = (
            df.groupby(group_cols, dropna=False)["delayed"]
              .apply(lambda s: _rolling_mean_shifted(s, window=w))
              .reset_index(level=group_cols, drop=True)
              .astype("float32")
        )
    return df

def temporal_balanced_sample_per_month(X: pd.DataFrame, y: pd.Series, month_col: str, max_per_month: int, random_state=42):
    tmp = X.copy()
    tmp["__y__"] = y.values
    if month_col not in tmp.columns:
        raise ValueError(f"month_col '{month_col}' not found in X.")

    parts = []
    for m, g in tmp.groupby(month_col, dropna=False):
        if len(g) <= max_per_month:
            parts.append(g)
            continue

        y_counts = g["__y__"].value_counts().to_dict()
        total = len(g)
        n_pos = int(max_per_month * (y_counts.get(1, 0) / total))
        n_neg = max_per_month - n_pos

        pos = g[g["__y__"] == 1].sample(n=min(n_pos, y_counts.get(1, 0)), random_state=random_state)
        neg = g[g["__y__"] == 0].sample(n=min(n_neg, y_counts.get(0, 0)), random_state=random_state)
        sampled = pd.concat([pos, neg], axis=0)

        if len(sampled) < max_per_month:
            remaining = g.drop(sampled.index)
            need = max_per_month - len(sampled)
            if need > 0 and len(remaining) > 0:
                sampled = pd.concat(
                    [sampled, remaining.sample(n=min(need, len(remaining)), random_state=random_state)],
                    axis=0
                )

        parts.append(sampled)

    out = pd.concat(parts, axis=0).sample(frac=1, random_state=random_state)
    y_s = out["__y__"].astype(int)
    X_s = out.drop(columns="__y__")
    return X_s, y_s

# --------------------------
# Calibration (Platt)
# --------------------------
def platt_fit(proba_val, y_val):
    Xv = np.asarray(proba_val).reshape(-1, 1)
    yv = np.asarray(y_val).astype(int)

    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    lr.fit(Xv, yv)
    return lr

def platt_predict(lr, proba):
    X = np.asarray(proba).reshape(-1, 1)
    return lr.predict_proba(X)[:, 1]

# --------------------------
# Threshold selection: precision constraint
# --------------------------
def optimize_threshold_precision_constraint(y_true, proba, min_precision=0.60, thresholds=None):
    """
    Choose threshold that MAXIMIZES recall subject to precision >= min_precision.
    Returns (best_thr, df, best_row, mode)
      mode = "constraint_satisfied" or "fallback_best_precision"
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.001)

    y_true = np.asarray(y_true).astype(int)
    rows = []

    for t in thresholds:
        pred = (proba >= t).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        rec  = recall_score(y_true, pred, zero_division=0)
        f1   = f1_score(y_true, pred, zero_division=0)
        rows.append([t, prec, rec, f1])

    df = pd.DataFrame(rows, columns=["threshold","precision","recall","f1"])

    ok = df[df["precision"] >= min_precision]
    if len(ok) > 0:
        best = ok.sort_values(["recall", "precision", "f1"], ascending=False).iloc[0]
        mode = "constraint_satisfied"
    else:
        best = df.sort_values(["precision", "recall"], ascending=False).iloc[0]
        mode = "fallback_best_precision"

    best_thr = float(best["threshold"])
    return best_thr, df, best, mode

def save_threshold_curve(df, best_thr, title, out_path: Path):
    plt.figure(figsize=(10,5))
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["f1"], label="F1")
    plt.axvline(best_thr, linestyle="--", label=f"Best thr={best_thr:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# --------------------------
# Load data
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

for c in ["YEAR","MONTH","DAY","DAY_OF_WEEK","CANCELLED","DIVERTED"]:
    flights[c] = pd.to_numeric(flights[c], errors="coerce").astype("Int16")

for c in ["SCHEDULED_DEPARTURE","SCHEDULED_ARRIVAL","SCHEDULED_TIME","DISTANCE","FLIGHT_NUMBER","ARRIVAL_DELAY"]:
    flights[c] = pd.to_numeric(flights[c], errors="coerce").astype("float32")

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
print("Delayed rate:", round(float(flights_model["delayed"].mean()), 4))

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

flights_model = add_congestion_features(flights_model)

# rolling target rates (no leakage)
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
    "origin_day_hour_logvol","dest_day_hour_logvol",
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

if not DROP_RAW_HHMM_COLS:
    base_features += ["SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL"]

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
print("Train delayed rate:", round(float(y_train.mean()), 4))
print("Val   delayed rate:", round(float(y_val.mean()), 4))
print("Test  delayed rate:", round(float(y_test.mean()), 4))

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
    cat_features.append("FLIGHT_NUMBER")

cat_features = [c for c in cat_features if c in X_train.columns]

X_train = sanitize_cat_cols(X_train, cat_features)
X_val   = sanitize_cat_cols(X_val, cat_features)
X_test  = sanitize_cat_cols(X_test, cat_features)

for df_ in (X_train, X_val, X_test):
    for c in df_.columns:
        if c not in cat_features:
            df_[c] = pd.to_numeric(df_[c], errors="coerce").astype("float32")

train_prior = float(y_train.mean())
num_cols = [c for c in X_train.columns if c not in cat_features]
for c in num_cols:
    X_train[c] = X_train[c].fillna(train_prior).astype("float32")
    X_val[c]   = X_val[c].fillna(train_prior).astype("float32")
    X_test[c]  = X_test[c].fillna(train_prior).astype("float32")

# --------------------------
# Train sampling (temporal-balanced)
# --------------------------
if USE_TRAIN_SAMPLING:
    X_train_s, y_train_s = temporal_balanced_sample_per_month(
        X_train, y_train,
        month_col="MONTH",
        max_per_month=MAX_SAMPLES_PER_MONTH,
        random_state=RANDOM_STATE
    )
    X_train_s = sanitize_cat_cols(X_train_s, cat_features)
    for c in X_train_s.columns:
        if c not in cat_features:
            X_train_s[c] = pd.to_numeric(X_train_s[c], errors="coerce").astype("float32").fillna(train_prior)

    X_train, y_train = X_train_s, y_train_s
    print(f"✅ Train size after temporal-balanced sampling: {X_train.shape} | delayed rate: {round(float(y_train.mean()), 4)}")
else:
    print("ℹ️ Train sampling disabled; using full TRAIN set.")

cat_idx = [X_train.columns.get_loc(c) for c in cat_features]

neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
base_spw = neg / max(pos, 1)
scale_pos_weight = base_spw * SPW_MULTIPLIER

print("scale_pos_weight (base):", round(base_spw, 4), "| multiplier:", SPW_MULTIPLIER, "| (used):", round(scale_pos_weight, 4))
print("Class balance:", {"neg": neg, "pos": pos, "scale_pos_weight": round(scale_pos_weight, 4)})

train_pool = Pool(X_train, y_train, cat_features=cat_idx)
val_pool   = Pool(X_val, y_val, cat_features=cat_idx)
test_pool  = Pool(X_test, y_test, cat_features=cat_idx)

cb = CatBoostClassifier(**CB_PARAMS, scale_pos_weight=scale_pos_weight)
cb.fit(train_pool, eval_set=val_pool, use_best_model=True)

# Base probabilities
proba_val_base  = cb.predict_proba(val_pool)[:, 1]
proba_test_base = cb.predict_proba(test_pool)[:, 1]

# Platt calibration on VAL
platt = platt_fit(proba_val_base, y_val)
proba_val  = platt_predict(platt, proba_val_base)
proba_test = platt_predict(platt, proba_test_base)

# Prior shift (VAL -> TEST)
pi_val = float(np.mean(y_val))
pi_test = float(np.mean(y_test))  # offline; in production estimate on recent window

print("pi_val :", round(pi_val, 4))
print("pi_test:", round(pi_test, 4))

# ============================================================
# Prior-shift adjustment (VAL -> TEST base-rate)
# ============================================================
proba_val_adj  = prior_shift_adjust(
    proba_val,
    pi_source=pi_val,
    pi_target=pi_test
)

proba_test_adj = prior_shift_adjust(
    proba_test,
    pi_source=pi_val,
    pi_target=pi_test
)

# ============================================================
# Evaluation @ threshold = 0.50
# ============================================================

# --- Calibrated only (NO prior-shift)
pred_05_cal = (proba_test >= 0.5).astype(int)

auc_cal = roc_auc_score(y_test, proba_test)
prec_cal = precision_score(y_test, pred_05_cal, zero_division=0)
rec_cal  = recall_score(y_test, pred_05_cal, zero_division=0)
f1_cal   = f1_score(y_test, pred_05_cal, zero_division=0)

# --- Calibrated + prior-shift
pred_05_adj = (proba_test_adj >= 0.5).astype(int)

auc_adj = roc_auc_score(y_test, proba_test_adj)
prec_adj = precision_score(y_test, pred_05_adj, zero_division=0)
rec_adj  = recall_score(y_test, pred_05_adj, zero_division=0)
f1_adj   = f1_score(y_test, pred_05_adj, zero_division=0)

print("\n==============================")
print("Model: CatBoost V6.2")
print("Calibration: Platt")
print("Prior-shift: VAL → TEST")
print("==============================")
print(f"pi_val  : {pi_val:.4f}")
print(f"pi_test : {pi_test:.4f}")

print("\n📌 TEST @ thr = 0.50 (calibrated)")
print(f"ROC-AUC  : {auc_cal:.4f}")
print(f"Precision: {prec_cal:.4f}")
print(f"Recall   : {rec_cal:.4f}")
print(f"F1       : {f1_cal:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, pred_05_cal))

print("\n📌 TEST @ thr = 0.50 (calibrated + prior-shift)")
print(f"ROC-AUC  : {auc_adj:.4f}")
print(f"Precision: {prec_adj:.4f}")
print(f"Recall   : {rec_adj:.4f}")
print(f"F1       : {f1_adj:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, pred_05_adj))

# ===========================================================
# Clssification report (calibrated only)
# ===========================================================
report_cal = classification_report(y_test, pred_05_cal, digits=4)
print("\n📋 Classification Report (calibrated only):\n", report_cal)

# ============================================================
# Classification report (calibrated + prior-shift)
# ============================================================
report_adj = classification_report(y_test, pred_05_adj, digits=4)
print("\n📋 Classification Report (calibrated + prior-shift):\n", report_adj)

# ============================================================
# ROC Curve (calibrated probabilities)
# ============================================================
RocCurveDisplay.from_predictions(y_test, proba_test)
plt.title("ROC Curve - CatBoost V6.2 (calibrated)")
roc_path = OUTPUT_DIR / "roc_curve_catboost_v62.png"
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n🖼️ Saved ROC curve to: {roc_path}")

# ============================================================
# Threshold selection
# Goal: maximize RECALL subject to Precision >= MIN_PRECISION_VAL
# IMPORTANT:
# - threshold is selected on VAL_ADJ
# - applied on TEST_ADJ
# ============================================================
print("\n==============================")
print(f"🎯 Threshold selection on VAL_ADJ")
print(f"Constraint: Precision >= {MIN_PRECISION_VAL:.2f}")
print("==============================")

best_thr, thr_df, best_row, mode = optimize_threshold_precision_constraint(
    y_val,
    proba_val_adj,
    min_precision=MIN_PRECISION_VAL
)

print("Mode:", mode)
print(f"✅ Best threshold: {best_thr:.6f}")
print(best_row)

thr_path = OUTPUT_DIR / f"threshold_curve_catboost_v62_prec{int(MIN_PRECISION_VAL*100)}.png"
save_threshold_curve(
    thr_df,
    best_thr,
    title=f"Threshold vs Precision / Recall / F1 (VAL_ADJ)",
    out_path=thr_path
)
print(f"🖼️ Saved threshold curve to: {thr_path}")

# ============================================================
# TEST metrics @ selected threshold (adjusted probs)
# ============================================================
pred_opt = (proba_test_adj >= best_thr).astype(int)

prec_opt = precision_score(y_test, pred_opt, zero_division=0)
rec_opt  = recall_score(y_test, pred_opt, zero_division=0)
f1_opt   = f1_score(y_test, pred_opt, zero_division=0)

print("\n📊 TEST metrics @ best threshold (chosen on VAL_ADJ)")
print(f"Threshold: {best_thr:.6f}")
print(f"Precision: {prec_opt:.4f}")
print(f"Recall   : {rec_opt:.4f}")
print(f"F1       : {f1_opt:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, pred_opt))

# ============================================================
# Save artifacts
# ============================================================
model_path = OUTPUT_DIR / "catboost_model_v62.cbm"
cb.save_model(str(model_path))
print(f"\n💾 Saved CatBoost model to: {model_path}")

platt_path = OUTPUT_DIR / "platt_calibrator_v62.joblib"
joblib.dump(platt, platt_path)
print(f"💾 Saved Platt calibrator to: {platt_path}")

thr_txt = OUTPUT_DIR / f"best_threshold_v62_prec{int(MIN_PRECISION_VAL*100)}.txt"
thr_txt.write_text(str(best_thr))
print(f"💾 Saved best threshold to: {thr_txt}")
