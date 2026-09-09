import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.pkl')
robust_scaler = joblib.load('robust_scaler.pkl')
standard_scaler = joblib.load('standard_scaler.pkl')
ohe = joblib.load('onehot_encoder.pkl')
defaults = joblib.load('defaults.pkl')
nominal_defaults = joblib.load('nominal_defaults.pkl')
remaining_numeric = joblib.load('remaining_numeric.pkl')

defaults.update(nominal_defaults)

quality_order = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
exposure_order = ['None', 'No', 'Mn', 'Av', 'Gd']
bsmt_fin_order = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
garage_finish_order = ['None', 'Unf', 'RFn', 'Fin']

nominal_cols = [
    'MSZoning',
    'Street',
    'LotShape',
    'LandContour',
    'Utilities',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'Foundation',
    'Heating',
    'CentralAir',
    'Electrical',
    'Functional',
    'GarageType',
    'PavedDrive',
    'SaleType',
    'SaleCondition',
    'Alley',
    'Fence',
    'MiscFeature'
]

skewed_cols = [
    'LotArea',
    'LotFrontage',
    'MasVnrArea',
    'TotalBsmtSF',
    '1stFlrSF',
    'GrLivArea'
]

ordinal_cols = [
    'ExterQual',
    'ExterCond',
    'BsmtQual',
    'BsmtCond',
    'HeatingQC',
    'KitchenQual',
    'FireplaceQu',
    'GarageQual',
    'GarageCond',
    'PoolQC'
]


def encode_ordinal(value, categories):
    if pd.isna(value):
        return 0.0

    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)

        if value.is_integer() and 0 <= value < len(categories):
            return value

    value = str(value).strip()

    if value.lower() == 'none':
        value = 'None'

    mapping = {
        'po': 'Po',
        'fa': 'Fa',
        'ta': 'TA',
        'gd': 'Gd',
        'ex': 'Ex',
        'no': 'No',
        'mn': 'Mn',
        'av': 'Av',
        'unf': 'Unf',
        'lwq': 'LwQ',
        'rec': 'Rec',
        'blq': 'BLQ',
        'alq': 'ALQ',
        'glq': 'GLQ',
        'rfn': 'RFn',
        'fin': 'Fin'
    }

    value = mapping.get(value.lower(), value)

    if value in categories:
        return float(categories.index(value))

    return 0.0


def preprocess_input(raw_input):
    row = defaults.copy()
    row.update(raw_input)

    df_input = pd.DataFrame([row])

    for col in ordinal_cols:
        df_input[col] = encode_ordinal(
            df_input[col].iloc[0],
            quality_order
        )

    df_input['BsmtExposure'] = encode_ordinal(
        df_input['BsmtExposure'].iloc[0],
        exposure_order
    )

    df_input['BsmtFinType1'] = encode_ordinal(
        df_input['BsmtFinType1'].iloc[0],
        bsmt_fin_order
    )

    df_input['BsmtFinType2'] = encode_ordinal(
        df_input['BsmtFinType2'].iloc[0],
        bsmt_fin_order
    )

    df_input['GarageFinish'] = encode_ordinal(
        df_input['GarageFinish'].iloc[0],
        garage_finish_order
    )

    for col in nominal_cols:
        if pd.isna(df_input[col].iloc[0]):
            df_input.loc[0, col] = defaults.get(col, 'None')

    df_input[nominal_cols] = df_input[nominal_cols].astype(str)

    encoded = ohe.transform(
        df_input[nominal_cols]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(nominal_cols),
        index=df_input.index
    )

    df_input = pd.concat(
        [
            df_input.drop(columns=nominal_cols),
            encoded_df
        ],
        axis=1
    )

    df_input = df_input.loc[
        :,
        ~df_input.columns.duplicated(keep='last')
    ]

    for col in skewed_cols:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(
                df_input[col],
                errors='coerce'
            )

    for col in remaining_numeric:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(
                df_input[col],
                errors='coerce'
            )

    for col in skewed_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].fillna(0)

    for col in remaining_numeric:
        if col in df_input.columns:
            df_input[col] = df_input[col].fillna(0)

    df_input[skewed_cols] = robust_scaler.transform(
        df_input[skewed_cols]
    )

    df_input[remaining_numeric] = standard_scaler.transform(
        df_input[remaining_numeric]
    )

    df_input = df_input.loc[
        :,
        ~df_input.columns.duplicated(keep='last')
    ]

    df_input = df_input.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    return df_input


st.title("House Price Predictor (Ames, USA)")
st.markdown("---")
st.caption("By Taha Momin")

overall_qual = st.slider(
    "Overall Quality (1-10)",
    1,
    10,
    5
)

gr_liv_area = st.number_input(
    "Living Area (sqft)",
    min_value=1,
    value=1500
)

neighborhood = st.selectbox(
    "Neighborhood",
    [
        'CollgCr',
        'Veenker',
        'Crawfor',
        'NoRidge',
        'Mitchel',
        'Somerst',
        'NWAmes',
        'OldTown',
        'BrkSide',
        'Sawyer',
        'NridgHt',
        'NAmes',
        'SawyerW',
        'IDOTRR',
        'MeadowV',
        'Edwards',
        'Timber',
        'Gilbert',
        'StoneBr',
        'ClearCr',
        'NPkVill',
        'Blmngtn',
        'BrDale',
        'SWISU',
        'Blueste'
    ],
    key="neighborhood"
)

total_bsmt_sf = st.number_input(
    "Basement Area (sqft)",
    min_value=0,
    value=800
)

garage_cars = st.slider(
    "Garage Capacity (cars)",
    0,
    4,
    2
)

year_built = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2000
)

full_bath = st.slider(
    "Full Bathrooms",
    0,
    4,
    2
)

tot_rms_abv_grd = st.slider(
    "Total Rooms Above Grade",
    2,
    14,
    6
)

lot_area = st.number_input(
    "Lot Area (sqft)",
    min_value=1,
    value=9000
)

fireplaces = st.slider(
    "Fireplaces",
    0,
    3,
    0
)

quality_labels = {
    'None': 'None',
    'Poor': 'Po',
    'Fair': 'Fa',
    'Average': 'TA',
    'Good': 'Gd',
    'Excellent': 'Ex'
}

kitchen_qual_display = st.selectbox(
    "Kitchen Quality",
    list(quality_labels.keys()),
    key="kitchen_quality"
)

kitchen_qual = quality_labels[kitchen_qual_display]

if st.button(
    "Predict Price",
    key="predict_price"
):
    raw_input = {
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area,
        'Neighborhood': neighborhood,
        'TotalBsmtSF': total_bsmt_sf,
        'GarageCars': garage_cars,
        'YearBuilt': year_built,
        'FullBath': full_bath,
        'TotRmsAbvGrd': tot_rms_abv_grd,
        'LotArea': lot_area,
        'Fireplaces': fireplaces,
        'KitchenQual': kitchen_qual
    }

    try:
        processed = preprocess_input(raw_input)

        prediction_log = model.predict(processed)[0]

        prediction_price = np.expm1(
            prediction_log
        )

        st.success(
            f"Predicted Sale Price: ${prediction_price:,.0f}"
        )

    except Exception as e:
        st.error(
            f"Prediction failed: {e}"
        )