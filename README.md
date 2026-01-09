# ✈️ ML delay flight predict

## Flight Delay Prediction in the United States

This repository presents a **complete Data Science and Machine Learning pipeline** for the problem of **flight delay classification**.

The project focuses on **experimental rigor**, **data leakage prevention**, **realistic evaluation**, and **critical interpretation of results**, going beyond simply training models.

---

## 🎯 Objective

Predict whether a flight will **be delayed or not**, using only information **available before the flight**, simulating a real-world production scenario.

---

## 📊 Dataset

* **Source:** Public U.S. flights dataset
* **Initial records:** ~5.8 million flights
* **After cleaning and merges:** ~5.7 million records
* **Target:** flight delay (binary classification)
* **Positive class (delay):** ~18–20%

> The dataset presents **moderate class imbalance**, typical of real-world problems, making accuracy insufficient when used alone.

---

## 🧹 Preprocessing and Feature Engineering

### Data Cleaning

* Removal of invalid or inconsistent records
* Appropriate handling of missing values

### Removal of Post-Flight Variables (Leakage Prevention)

Variables known **only after the flight occurs** were explicitly removed, such as:

* actual delays
* final operational times
* arrival and departure outcome metrics

> Keeping these variables would cause **direct data leakage**, artificially inflating model performance.

### Temporal and Historical Features

* Temporal variables (month, hour, day of week)
* Historical statistics by airport, airline, and route

#### Historical Features Toggle

```text
USE_OPERATIONAL_DELAY_COLS = True
```

* Operational variables **do not enter the model directly**
* They are used only to **build historical aggregated statistics**
* A **temporal shift** is applied: for a flight in month T, only data up to T−1 is used

> This ensures **causality** and **zero data leakage**.

---

## ⏱️ Data Splitting Strategy

A **temporal split** was adopted to simulate future prediction:

| Dataset    | Months |
| ---------- | ------ |
| Train      | 1–8    |
| Validation | 9–10   |
| Test       | 11–12  |

* The **test set preserves the real-world distribution**
* Class balancing was applied **only to the training set**

> Temporal splitting prevents information leakage and ensures realistic evaluation.

---

## ⚖️ Class Imbalance

* Delays represent only ~20% of the data
* A naive classifier predicting “no delay” would achieve ~80% accuracy

### Design Decisions

* Use of robust metrics: **ROC-AUC, Precision, Recall, F1-score**
* Adjustment of `scale_pos_weight`
* Conscious threshold optimization
* Precision target

---

## 🤖 Models Evaluated

### Baselines

* **Logistic Regression** (interpretable linear baseline)
* **SGDClassifier (log-loss)** – discarded due to instability

### Advanced Model

* **CatBoost (Gradient Boosting)**

  * Captures non-linear relationships
  * Highly effective for tabular data

---

## 📏 Evaluation Metrics

* **ROC-AUC** – separation capability
* **Precision / Recall / F1-score**
* **Confusion Matrix**

> Accuracy is reported only as a secondary metric due to class imbalance.

---

## 📈 Main Results

### Logistic Regression (Baseline)

* ROC-AUC: **0.63**
* High recall but low precision
* Clear limitation in capturing complex patterns

### CatBoost (Final Model)

* ROC-AUC: **~0.83**
* Superior separation and better balance across metrics

> The significant ROC-AUC improvement highlights the importance of non-linear models for this problem.

---

## 🎯 Probability Calibration (Platt Scaling)

Despite strong discriminative performance, CatBoost exhibited poorly calibrated probabilities.

### Solution

* **Platt Scaling** using **Logistic Regression** as a calibrator

Workflow:

1. Train CatBoost
2. Generate probabilities on the validation set
3. Train the Platt calibrator
4. Apply calibration to the test set

> Calibration does not change ROC-AUC but makes probabilities interpretable and reliable.

---

## 🔁 Prior-Shift Adjustment

A difference in delay prevalence was observed between validation and test sets:

```text
pi_val  = 12.7%
pi_test = 17.9%
```

### Adjustment Applied

* **Prior-shift correction** applied after calibration
* Adjustment performed on **probabilities**, not on data samples

Observed effect:

* Recall ↑
* Precision ↓
* ROC-AUC unchanged

> The model correctly reflects the new statistical context of the test period.

---

## 🔎 Threshold Selection

* Threshold not chosen arbitrarily
* Selected on the validation set with the constraint:

```text
Precision ≥ 0.66
```

Result:

* Final threshold ≈ **~0.61**
* Controlled trade-off between false positives and false negatives

> This choice reflects a realistic operational requirement.

---

## 📊 Generated Visualizations

* ROC Curve (baseline × CatBoost)
* Precision × Recall / Threshold Curve
* Confusion matrices (default and optimized thresholds)
* Calibration curves (before × after)
* Score distribution by class
* Feature importance (CatBoost)

---

## 🧠 Key Conclusions

* Linear models are useful baselines but insufficient
* CatBoost significantly outperforms linear approaches
* Calibration and prior-shift are essential for real-world usage
* A portion of delays is structural and historically predictable


## 🏁 Final Remarks

This project demonstrates a **Machine Learning pipeline**, emphasizing statistical validity, interpretability, and a try of real-world applicability.
