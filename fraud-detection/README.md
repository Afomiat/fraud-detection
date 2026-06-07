# Fraud Detection for E-Commerce and Credit Card Transactions

An end-to-end machine learning system designed to detect fraudulent transactions using advanced exploratory data analysis (EDA), feature engineering, and robust classification models. This project utilizes two distinct datasets: an e-commerce transaction dataset coupled with IP-to-country mapping, and a highly imbalanced European credit card transactions dataset.

---

## 🚀 Project Overview

The goal of this project is to identify patterns of fraudulent behavior and build predictive models to flag suspicious transactions. The repository is structured to support clean, reproducible research and seamless deployment via CI/CD pipelines.

### Key Milestones & Completed Work

1. **Repository Setup**: Initialized directory structure, configured Git tracking on the `task-1` branch, and set up `.gitignore` to preserve folder hierarchies (`data/raw/`, `data/processed/`) without tracking large datasets.
2. **CI/CD Integration**: Configured automated workflows with GitHub Actions (`.github/workflows/unittests.yml`) to run unit tests automatically on push/pull requests.
3. **Exploratory Data Analysis (EDA)**:
   - In-depth, step-by-step EDA for both e-commerce and credit card datasets with detailed markdown insights directly above each code execution.
   - Identified data quality issues, handled datetime conversions, resolved class imbalances, and mapped IP addresses to countries.
4. **Unit Testing**: Implemented a testing suite (`tests/`) integrated with pytest to ensure pipeline reliability.

---

## 📂 Project Structure

```text
fraud-detection/
├── .github/
│   └── workflows/
│       └── unittests.yml      # GitHub Actions CI/CD configuration
├── .vscode/
│   └── settings.json          # Workspace settings
├── data/
│   ├── raw/                   # Raw source datasets (gitignored)
│   └── processed/             # Cleaned and processed features (gitignored)
├── models/                    # Serialized machine learning models (gitignored)
├── notebooks/
│   ├── eda-creditcard.ipynb   # EDA and preprocessing for credit card data
│   └── eda-fraud-data.ipynb   # EDA, IP mapping, and engineering for e-commerce data
├── scripts/                   # Executable python modules & utility scripts
├── src/                       # Core package source code
├── tests/
│   └── test_placeholder.py    # Unit tests for CI/CD verification
├── .gitignore                 # Excludes data, virtual environments, and caches
├── README.md                  # Project documentation (this file)
└── requirements.txt           # Python dependencies
```

---

## 🔍 Exploratory Data Analysis & Insights

### 1. E-Commerce Fraud Dataset (`eda-fraud-data.ipynb`)
This analysis focuses on user behavior and metadata to distinguish legitimate transactions from fraudulent ones.

* **Dataset Profile**: Contains 151,112 rows and 11 columns covering user profiles, sign-up times, and transaction values.
* **Data Quality**: Verified 0 missing values and 0 duplicate rows. Converted string-based `signup_time` and `purchase_time` to datetime format.
* **Feature Engineering & Behavioral Insights**:
  * **IP-to-Country Mapping**: Successfully mapped integer-format IP addresses to their corresponding countries of origin. Analyzed fraud rates by country to identify high-risk jurisdictions.
  * **Velocity/Time Analysis**: Discovered that a significant portion of fraud occurs almost instantly after registration. Calculated `time_diff` (time between signup and purchase) and flagged transactions with near-zero latency.
  * **Device Re-use**: Identified devices associated with multiple user accounts. Multiple sign-ups from the same device ID are a strong indicator of automated bot/syndicate fraud.
  * **Temporal Patterns**: Analyzed hourly, daily, and weekly fraud distributions to discover peak periods of malicious activity.

### 2. Credit Card Fraud Dataset (`eda-creditcard.ipynb`)
A benchmark dataset containing 284,807 European card transactions, characterized by extreme class imbalance.

* **Dataset Profile**: Features are anonymized numerical columns obtained via PCA projection (V1–V28), plus `Time`, `Amount`, and the target `Class`.
* **Data Quality**: Identified and dropped 1,081 duplicate transactions.
* **Class Imbalance**: Fraud transactions represent only **0.17%** (492 cases) of the entire dataset. Acknowledged that standard accuracy is a misleading metric; optimized for **Area Under the Precision-Recall Curve (AUPRC)** and **F1-Score**.
* **Preprocessing Pipeline**:
  * Scaled the non-PCA columns (`Time` and `Amount`) to match the distribution of the PCA features.
  * Split the dataset into train and test sets **before** performing any oversampling to eliminate data leakage.
  * Applied **SMOTE (Synthetic Minority Over-sampling Technique)** exclusively to the training set to balance the target class, while keeping the validation test set in its raw, imbalanced state to simulate real-world production data.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- Git

### Local Environment Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd fraud-detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Running Tests & CI/CD

To run the test suite locally:
```bash
python -m pytest tests/ -v
```

GitHub Actions automatically executes the unit test pipeline on every push to `main` and `task-1`, ensuring all changes meet the project's quality standard before merging.
