### feature_engineering.py
import numpy as np
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

def transform_cyclical_features(data):
    months_in_a_year = 12
    data['sin_MoSold'] = np.sin(2 * np.pi * (data['MoSold'] - 1) / months_in_a_year)
    data['cos_MoSold'] = np.cos(2 * np.pi * (data['MoSold'] - 1) / months_in_a_year)
    return data.drop(columns=['MoSold'])

def prepare_pipeline(X_train):
    numerical_features = X_train.select_dtypes(include=["float64", "int64"]).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    ordinal_features = ["OverallQual", "ExterQual", "BsmtQual", "BsmtExposure", "BsmtFinType1",
                        "HeatingQC", "KitchenQual", "FireplaceQu", "GarageFinish", "GarageQual",
                        "GarageCond", "PoolQC", "Fence"]

    encoder_ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    preproc_ordinal = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"), encoder_ordinal, MinMaxScaler()
    )

    preproc_nominal = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )

    preproc_numerical = make_pipeline(KNNImputer(), MinMaxScaler())

    return make_column_transformer(
        (preproc_numerical, numerical_features),
        (preproc_nominal, [col for col in categorical_features if col not in ordinal_features]),
        (preproc_ordinal, ordinal_features),
        remainder="drop"
    )
