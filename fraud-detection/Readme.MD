# Fraud Detection — E-commerce & Bank Transactions
**Adey Innovations Inc. | 10 Academy Week 5–6 Challenge**

A end-to-end machine learning pipeline for detecting fraudulent transactions across two distinct datasets: e-commerce transactions with rich behavioral context, and bank credit card transactions with PCA-anonymized features.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Task 1 — Data Analysis & Preprocessing](#task-1--data-analysis--preprocessing)
- [Task 2 — Model Building & Training](#task-2--model-building--training)
- [Task 3 — Model Explainability (SHAP)](#task-3--model-explainability-shap)
- [Key Findings](#key-findings)
- [Business Recommendations](#business-recommendations)
- [Results Summary](#results-summary)

---

## Project Overview

Effective fraud detection has direct financial and reputational consequences. False positives — flagging legitimate transactions as fraud — frustrate customers and erode trust. False negatives — missing actual fraud — drive direct financial loss.

This project builds a unified fraud detection system that:
1. Cleans, enriches, and engineers features from raw transaction data
2. Trains and compares multiple classification models on highly imbalanced data
3. Interprets the best model using SHAP to produce actionable business recommendations

---

## Project Structure

```
fraud-detection/
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/
│   ├── raw/                        # Original datasets (gitignored)
│   └── processed/                  # Cleaned and feature-engineered data
├── notebooks/
│   ├── eda-fraud-data.ipynb        # EDA for Fraud_Data.csv
│   ├── eda-creditcard.ipynb        # EDA for creditcard.csv
│   ├── modeling.ipynb              # Model training — Fraud_Data.csv
│   ├── modeling-creditcard.ipynb   # Model training — creditcard.csv
│   └── shap-explainability.ipynb   # SHAP analysis — both datasets
├── src/
│   ├── __init__.py
│   └── evaluate.py                 # Reusable model evaluation function
├── models/                         # Saved model artifacts (.pkl files)
├── scripts/
│   └── __init__.py
├── tests/
│   └── __init__.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Datasets

| Dataset | Rows | Features | Fraud Rate |
|---|---|---|---|
| `Fraud_Data.csv` | 151,112 | 11 | 9.4% |
| `IpAddress_to_Country.csv` | 138,846 | 3 | — |
| `creditcard.csv` | 284,807 | 31 | 0.17% |

**Fraud_Data.csv** contains e-commerce transactions with user, device, and behavioral context (signup time, IP address, browser, device ID).

**creditcard.csv** contains bank credit card transactions where features V1–V28 have been anonymized via PCA for privacy. Only `Time` and `Amount` are in original form.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
jupyter
ipykernel
joblib
```

> **Note:** Place raw datasets in `data/raw/` before running any notebooks.

---

## Task 1 — Data Analysis & Preprocessing

**Notebook:** `notebooks/eda-fraud-data.ipynb` and `notebooks/eda-creditcard.ipynb`

### What the Task Asked
- Clean both datasets (handle missing values, duplicates, fix data types)
- Perform exploratory data analysis (EDA) on distributions and fraud patterns
- Enrich e-commerce data with country information via IP geolocation
- Engineer behavioral features capturing fraud patterns
- Scale and encode features
- Handle class imbalance using SMOTE

### What We Found (EDA)

**Class Imbalance:**
- Fraud_Data.csv: 90.6% legitimate vs 9.4% fraud
- creditcard.csv: 99.83% legitimate vs 0.17% fraud — extreme imbalance

**Key patterns discovered:**
- Purchase value and age showed almost no difference between fraud and legitimate transactions — weak signals
- Direct traffic source had the highest fraud rate at 10.5% vs SEO at 8.9%
- Fraud peaked at 9am and 5pm (business hours) — not overnight as commonly assumed
- Fraud dipped Tuesday–Wednesday and peaked Friday–Sunday

### How We Did It — Geolocation (IP-to-Country Merge)

IP addresses cannot be joined to country ranges with a standard join because the mapping file stores IP ranges, not individual values. We used `pandas.merge_asof` — a range-based lookup that finds the closest lower bound for each IP:

```python
df_merged = pd.merge_asof(
    df_sorted,
    ip_df_sorted[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
    left_on='ip_int',
    right_on='lower_bound_ip_address',
    direction='backward'
)
# IPs outside any range → assigned 'Unknown' (kept, not dropped)
```

**Result:** 85.9% of IPs mapped to a country. 14.1% assigned 'Unknown'. All 151,112 rows preserved.

**Top fraud countries (min. 200 transactions):**

| Country | Fraud Rate |
|---|---|
| Ireland | 22.92% |
| New Zealand | 22.30% |
| Saudi Arabia | 18.94% |
| Denmark | 15.92% |
| Chile | 15.35% |

### How We Did It — Feature Engineering

Three new features were created from raw columns:

**`time_since_signup`** — seconds between `signup_time` and `purchase_time`:
```python
df['time_since_signup'] = (
    df['purchase_time'] - df['signup_time']
).dt.total_seconds()
```
- Fraud median: **1 second** after signup
- Legitimate median: **~60 days** after signup
- Most powerful feature in the entire dataset

**`hour_of_day` and `day_of_week`** — extracted from `purchase_time`:
```python
df['hour_of_day'] = df['purchase_time'].dt.hour
df['day_of_week'] = df['purchase_time'].dt.dayofweek
```

**`device_tx_count_1hr`** — number of transactions from the same device in the past hour:
- Fraud average: **2.68 transactions/hour**, max 18
- Legitimate average: **~0.000015** (virtually never reused)

### How We Did It — Data Transformation

```python
# One-hot encode categoricals
df_model = pd.get_dummies(df_model, columns=['source','browser','sex','country'])
# Fraud_Data: 11 columns → 198 columns after encoding

# Stratified train/test split (preserves fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features — fit on TRAIN only to prevent data leakage
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols]  = scaler.transform(X_test[numerical_cols])
```

> **Why fit on train only?** Fitting the scaler on all data lets it "see" the test set statistics — a form of data leakage that makes test performance look artificially better.

### How We Did It — Handling Class Imbalance (SMOTE)

SMOTE (Synthetic Minority Oversampling Technique) was chosen over undersampling because undersampling would discard ~100,000 legitimate rows, losing valuable signal. SMOTE creates synthetic fraud examples by interpolating between real ones.

**Critical rule: SMOTE applied to training data only. Test data keeps real-world distribution.**

```python
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

| Dataset | Before SMOTE | After SMOTE | Rows Added |
|---|---|---|---|
| Fraud_Data.csv | 9.4% fraud | 50% / 50% | 98,247 synthetic |
| creditcard.csv | 0.17% fraud | 50% / 50% | 227,057 synthetic |

**Processed files saved to `data/processed/`:**
- `X_train_fraud.csv`, `X_test_fraud.csv`, `y_train_fraud.csv`, `y_test_fraud.csv`
- `X_train_creditcard.csv`, `X_test_creditcard.csv`, `y_train_creditcard.csv`, `y_test_creditcard.csv`

---

## Task 2 — Model Building & Training

**Notebooks:** `notebooks/modeling.ipynb` and `notebooks/modeling-creditcard.ipynb`

### What the Task Asked
- Train a Logistic Regression baseline model
- Train ensemble models (XGBoost and LightGBM) with basic hyperparameter tuning
- Evaluate using metrics appropriate for imbalanced data: AUC-PR, F1, Confusion Matrix
- Run Stratified K-Fold cross-validation (k=5)
- Select the best model with written justification

### Why Not Accuracy?

A model that always predicts "legitimate" achieves 90.6% accuracy on Fraud_Data.csv while catching **zero fraud**. We use:

- **AUC-PR** (primary) — most informative for imbalanced datasets
- **F1-Score** — penalises models that sacrifice either precision or recall
- **Recall** — fraction of actual fraud cases caught (missing fraud = financial loss)
- **Precision** — fraction of fraud flags that are real (false alarms = customer frustration)

### How We Did It

All three models trained on SMOTE-balanced training data, evaluated on real-world imbalanced test data:

```python
# Logistic Regression baseline
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
```

### Cross-Validation — Important Note on SMOTE Leakage

A naive CV approach (cross-validating on already-SMOTE'd data) produced artificially inflated scores (F1 = 0.97) because synthetic rows from training folds leaked into validation folds. The correct approach applies SMOTE **inside each fold**:

```python
for fold, (train_idx, val_idx) in enumerate(cv.split(X_np, y_np)):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        X_np[train_idx], y_np[train_idx]   # SMOTE on fold train only
    )
    model.fit(X_resampled, y_resampled)
    # validate on real unseen fold data
```

### Results — Fraud_Data.csv

| Model | AUC-PR | AUC-ROC | F1 | Precision | Recall | Fraud Caught |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5836 | 0.7469 | 0.6424 | 1.0000 | 0.4731 | 1,339 / 2,830 |
| XGBoost | 0.6202 | 0.7653 | 0.6837 | 0.9652 | 0.5293 | 1,498 / 2,830 |
| **LightGBM** ✅ | **0.6212** | **0.7650** | **0.6913** | **1.0000** | **0.5283** | **1,495 / 2,830** |

**Winner: LightGBM** — highest AUC-PR (primary metric), highest F1, perfect precision (zero false alarms), most stable CV performance (std=0.0053).

### Results — creditcard.csv

| Model | AUC-PR | AUC-ROC | F1 | Precision | Recall | Fraud Caught |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7249 | 0.9698 | 0.1094 | 0.0581 | 0.9184 | 90 / 98 |
| **XGBoost** ✅ | **0.8786** | **0.9799** | **0.8056** | **0.7373** | **0.8878** | **87 / 98** |
| LightGBM | 0.8453 | 0.9614 | 0.7757 | 0.7155 | 0.8469 | 83 / 98 |

**Winner: XGBoost** — highest AUC-PR (0.8786), F1 (0.8056), and AUC-ROC (0.9799). Note: Logistic Regression had high recall (91.8%) but near-zero precision — it flagged almost everything as fraud.

> The different winners per dataset reflect the different imbalance levels: LightGBM handles moderate imbalance (9.4%) better while XGBoost handles extreme imbalance (0.17%) better.

**Models saved to `models/`:**
```
models/
├── lightgbm_fraud.pkl
├── xgboost_fraud.pkl
├── logistic_regression_fraud.pkl
├── feature_names_fraud.pkl
├── lightgbm_creditcard.pkl
├── xgboost_creditcard.pkl
├── logistic_regression_creditcard.pkl
└── feature_names_creditcard.pkl
```

---

## Task 3 — Model Explainability (SHAP)

**Notebook:** `notebooks/shap-explainability.ipynb`

### What the Task Asked
- Extract built-in feature importance from the ensemble model
- Generate SHAP Summary Plot (global feature importance)
- Generate SHAP Force Plots for a True Positive, False Positive, and False Negative
- Identify top 5 fraud drivers
- Provide at least 3 actionable business recommendations

### What Is SHAP?

SHAP (SHapley Additive exPlanations) explains individual predictions by calculating each feature's contribution. For any single transaction:

```
Base rate (avg prediction):    +0.09
time_since_signup = 1 sec:     +0.45  ← pushes toward fraud
device_tx_count = 12:          +0.28  ← pushes toward fraud
country = Ireland:             +0.08  ← moderate push
purchase_value = $35:          -0.02  ← pushes toward legit
                               ──────
Final fraud probability:        0.88
```

```python
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values[1], X_sample, plot_type="bar")
```

### Top 5 Fraud Drivers — Fraud_Data.csv (LightGBM)

| Rank | Feature | Mean SHAP | Direction |
|---|---|---|---|
| 1 | `time_since_signup` | 2.40 | Low value → fraud (signed up recently) |
| 2 | `device_tx_count_1hr` | 1.00 | High value → fraud (many transactions) |
| 3 | `hour_of_day` | 0.80 | Peaks at 9am and 5pm |
| 4 | `day_of_week` | 0.52 | Weekends slightly higher risk |
| 5 | `country_United States` | 0.45 | Complex interaction with other features |

### Top 5 Fraud Drivers — creditcard.csv (XGBoost)

| Rank | Feature | Mean SHAP | Direction |
|---|---|---|---|
| 1 | `V14` | 1.90 | Low value → fraud |
| 2 | `V4` | 1.75 | High value → fraud |
| 3 | `V8` | 0.83 | Complex |
| 4 | `V12` | 0.80 | Complex |
| 5 | `V1` | 0.78 | Complex |

### Force Plot Analysis

**True Positive (Fraud Correctly Caught):**
The model caught this fraud because `time_since_signup` was near zero (+5.76 SHAP) and `device_tx_count_1hr` was very high (+3.43 SHAP). Two simultaneous smoking guns produced a high-confidence correct catch.

**False Positive (Legit Wrongly Flagged — XGBoost):**
A legitimate customer was flagged because they signed up relatively recently (+2.86 SHAP) AND transacted from Japan (+1.17 SHAP). No single feature was overwhelming — an unfortunate combination of mild signals pushed the prediction over the threshold.

**False Negative (Fraud Incorrectly Missed):**
A sophisticated fraudster evaded detection by mimicking normal user behavior — normal signup time (-2.32 SHAP), low device velocity (-1.01 SHAP), transacting at a normal hour (-0.96 SHAP). The model had no strong signal to work with.

> **Note on False Positives:** LightGBM achieved perfect precision on Fraud_Data (zero false positives). The FP example above uses XGBoost which had 54 false positives — illustrating the precision/recall tradeoff between the two models.

---

## Key Findings

1. **time_since_signup is the strongest fraud signal** — fraud median is 1 second after signup vs 60 days for legitimate users. This single feature has a SHAP value of 2.4 — more than double the next feature.

2. **Transaction velocity reveals automated fraud** — fraudsters average 2.68 device transactions per hour (max 18) vs essentially 0 for legitimate users. This captures card-testing attacks directly.

3. **Geographic patterns matter** — Ireland (22.9%), New Zealand (22.3%), and Saudi Arabia (18.9%) have fraud rates more than double the average. Country is a useful contextual risk signal.

4. **Fraud happens during business hours** — contrary to assumptions, fraud peaks at 9am and 5pm, not overnight. Fraudsters mimic normal user activity windows.

5. **Sophisticated fraudsters are invisible** — the False Negative analysis shows that fraudsters who mimic normal behavior (normal signup time, low velocity, average amounts) are extremely difficult to catch with behavioral features alone.

6. **V14 dominates bank fraud** — in the anonymized creditcard dataset, PCA feature V14 is by far the strongest signal (SHAP 1.90), appearing prominently in all three force plots.

7. **Model choice depends on imbalance level** — LightGBM outperforms on moderate imbalance (9.4%), XGBoost outperforms on extreme imbalance (0.17%).

---

## Business Recommendations

### 1. Signup-to-Purchase Time Gate
**SHAP evidence:** `time_since_signup` is the single most powerful feature with SHAP value 2.4. Fraud median = 1 second, legitimate median = 60 days.

**Action:** Any transaction occurring within 1 hour of account creation should automatically trigger secondary verification (email OTP, SMS confirmation, or manual review queue). This targets the majority of fraud cases with minimal disruption to real customers who typically buy days or weeks after signup.

---

### 2. Device Velocity Throttling
**SHAP evidence:** `device_tx_count_1hr` is the second strongest feature. Fraudsters average 2.68 transactions/hour from the same device — some devices made 18 transactions in a single hour. Legitimate users average essentially zero.

**Action:** Implement a hard cap of 3 transactions per device per hour. Transactions beyond this threshold should be automatically held for review. This directly counters card-testing attacks where fraudsters rapidly cycle through stolen card numbers.

---

### 3. Geographic Risk Scoring
**SHAP evidence:** Country features appear in the top 15 SHAP features. Ireland (22.9%), New Zealand (22.3%), and Saudi Arabia (18.9%) have fraud rates more than double the global average.

**Action:** Implement tiered geographic risk scoring. A new account transacting from a high-risk country should require additional verification. Flag combinations — recent signup AND high-risk country — as highest priority for review.

---

### 4. Graduated Response Over Binary Block
**SHAP evidence:** False Positive analysis shows legitimate customers get wrongly flagged due to combinations of mild signals (recent signup + foreign country). Hard blocking harms innocent customers.

**Action:** Replace binary block/allow with a three-tier response:
- **Low suspicion** → silent monitoring only
- **Medium suspicion** → step-up verification (OTP)
- **High suspicion** → transaction hold for manual review

This preserves customer experience for borderline cases while catching clear fraud.

---

### 5. Real-Time Temporal Monitoring
**SHAP evidence:** `hour_of_day` (SHAP 0.80) and `day_of_week` (SHAP 0.52) are consistently in the top 5 features. Fraud peaks at 9am and 5pm, dips Tuesday–Wednesday.

**Action:** Build a real-time dashboard tracking fraud rate by hour and day of week. Automated alerts should fire when any hour's fraud rate exceeds 1.5× the baseline. This enables the fraud team to respond proactively to emerging attack waves rather than discovering them in end-of-day reports.

---

## Results Summary

| Dataset | Best Model | AUC-PR | F1 | Fraud Caught | False Alarms |
|---|---|---|---|---|---|
| Fraud_Data.csv | LightGBM | 0.6212 | 0.6913 | 1,495 / 2,830 (52.8%) | 0 |
| creditcard.csv | XGBoost | 0.8786 | 0.8056 | 87 / 98 (88.8%) | 7 |

The credit card model performs significantly better because PCA features V1–V28 already encode rich fraud patterns extracted by the bank's data scientists. The e-commerce model's moderate performance reflects the inherent difficulty of catching sophisticated fraudsters who deliberately mimic legitimate behavior.

---

## Tech Stack

| Category | Tool | Purpose |
|---|---|---|
| Data processing | pandas, numpy | Loading, cleaning, merging, aggregation |
| Geolocation | merge_asof | Range-based IP-to-country lookup |
| Encoding | pd.get_dummies | One-hot encoding (11 → 198 columns) |
| Scaling | StandardScaler | Normalize features, fit on train only |
| Imbalance | SMOTE (imblearn) | Synthetic oversampling on training set only |
| Modeling | scikit-learn, XGBoost, LightGBM | Baseline + ensemble models |
| Explainability | SHAP | Feature importance + individual explanations |
| Visualization | matplotlib, seaborn | EDA plots, SHAP plots |
| Persistence | joblib | Save/load trained models |
| Environment | Python 3.x, VS Code, venv | Development environment |

---

*Adey Innovations Inc. — Fraud Detection Analytics | 10 Academy Week 5–6 | June 2026*