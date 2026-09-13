# Ames House Price Prediction

An end-to-end machine learning project that predicts residential home sale prices in Ames, Iowa, using the Ames Housing dataset. Covers the full pipeline: exploratory data analysis, data cleaning, feature engineering, model comparison, hyperparameter tuning, and deployment via a Streamlit web app.

## Problem

Given structural, quality, and location details about a house, predict its sale price.

## Dataset

- **Source:** [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) (via Kaggle / De Cock, 2011)
- **Size:** 1,460 rows, 81 columns (79 features + target `SalePrice`)
- **Target:** `SalePrice` (log-transformed during modeling to correct right-skew)

## Approach

### 1. Exploratory Data Analysis
- Checked target distribution — found strong right-skew (skew = 1.88), addressed with a `log1p` transform
- Identified 19 columns with missing values, split into two categories:
  - **"Doesn't apply"** columns (e.g. `PoolQC`, `GarageType`) — missing meant the feature doesn't exist for that house, filled with `"None"`/`0`
  - **Genuinely missing** columns (e.g. `LotFrontage`) — imputed using neighborhood-grouped medians
- Ran skew analysis across all numeric columns; separated genuinely skewed continuous features (e.g. `LotArea`) from "rare-event" zero-heavy columns (e.g. `PoolArea`, 99.5% zero), which were engineered into binary flags instead of log-transformed
- Correlation analysis and heatmaps identified `OverallQual` (0.79) and `GrLivArea` (0.71) as the strongest predictors
- Scatter plots revealed two outlier houses (>4,500 sqft living area, anomalously low price) — removed before modeling
- VIF analysis confirmed moderate multicollinearity (max VIF ≈ 5.3), not severe enough to require aggressive feature dropping

### 2. Preprocessing
- **Ordinal encoding** for quality-scale columns (`ExterQual`, `KitchenQual`, etc. on a Poor→Excellent scale; `BsmtExposure`, `BsmtFinType1/2`, `GarageFinish` on their own scales)
- **One-Hot encoding** for nominal categorical columns (`Neighborhood`, `Foundation`, `SaleType`, etc.)
- **RobustScaler** for genuinely skewed continuous features (resistant to outliers)
- **StandardScaler** for remaining continuous numeric features
- All scalers/encoders fit on the training set only, applied to test set via `.transform()` to avoid data leakage

### 3. Modeling

Four models were trained and compared:

| Model | RMSE (log scale) | R² |
|---|---|---|
| Linear Regression | 0.1176 | 0.9209 |
| Random Forest (tuned via GridSearchCV) | 0.1347 | 0.8963 |
| **ElasticNet (tuned via GridSearchCV)** | **0.1081** | **0.9333** |

**Final model: ElasticNet** (alpha=0.005, l1_ratio =0.1), selected for the best RMSE/R² after hyperparameter tuning. Interestingly, the tuned Random Forest underperformed both linear models — likely because the log-transformed target and heavily One-Hot-encoded feature space (200+ mostly-binary columns) suited linear models better than tree splits.

### 4. Deployment

A [Streamlit](https://streamlit.io/) web app (`app.py`) provides an interactive interface: users input key house details (quality rating, living area, neighborhood, etc.), and the app applies the same preprocessing pipeline before returning a predicted sale price.

## Project Structure

```
ames-house-price-prediction/
├── house_price_model.py      # EDA, cleaning, preprocessing, modeling
├── app.py                    # Streamlit deployment app
├── model.pkl                 # Final trained ElasticNet model
├── robust_scaler.pkl         # Fitted RobustScaler
├── standard_scaler.pkl       # Fitted StandardScaler
├── onehot_encoder.pkl        # Fitted OneHotEncoder
├── feature_defaults.pkl      # Median values for numeric features (form defaults)
├── nominal_defaults.pkl      # Mode values for categorical features (form defaults)
├── requirements.txt
└── README.md
```

## How to Run

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the Streamlit app:**
```bash
streamlit run app.py
```

**3. (Optional) Retrain the model from scratch:**
```bash
python house_price_model.py
```

## Key Techniques Used

- Missing value analysis (structural vs. genuine missingness)
- Skew detection and targeted transforms (log-transform vs. binary flagging)
- Ordinal vs. nominal encoding decisions based on category semantics
- Multicollinearity diagnosis via VIF (beyond pairwise correlation)
- Train/test-safe scaling (no data leakage)
- Hyperparameter tuning via `GridSearchCV`
- Model persistence and deployment with `joblib` + `Streamlit`

## Author

Taha Momin
