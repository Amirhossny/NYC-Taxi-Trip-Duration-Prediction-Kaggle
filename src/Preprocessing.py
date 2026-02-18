from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np

def remove_outliers(df,target_col ,factor=1.5):
    filtered_df = df.copy()
    Q1 = filtered_df[target_col].quantile(0.25)
    Q3 = filtered_df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    filtered_df = filtered_df[(filtered_df[target_col] >= lower ) & (filtered_df[target_col] <= upper )] 

    return filtered_df

def data_cleaning(df):
    min_lat, max_lat = 40.49, 40.92
    min_lon, max_lon = -74.27, -73.68
    df = df[(df['pickup_latitude'].between(min_lat, max_lat)) &
            (df['pickup_longitude'].between(min_lon, max_lon)) &
            (df['dropoff_latitude'].between(min_lat, max_lat)) &
            (df['dropoff_longitude'].between(min_lon, max_lon))]
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 7)]
    return df

# --- split features & target ---
def split_features_target(df, target_col):
    df = df.drop(columns=["id"], errors="ignore")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

# --- target preprocessing ---
def preprocess_target(y, use_log=True):
    if use_log:
        return np.log1p(y)
    return y

# --- inverse transform target ---
def inverse_target(y_transformed, use_log=True):
    if use_log:
        return np.expm1(y_transformed)
    return y_transformed


# --- binary encoding (store_and_fwd_flag) ---
def encode_store_flag(df):
    df = df.copy()
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].fillna("N").map({"Y":1,"N":0})
    return df[['store_and_fwd_flag']]

def _filter_existing_columns(config_cols, df_columns):
    return [c for c in config_cols if c in df_columns]

# --- build preprocessor pipeline ---

def build_preprocessor(pre_conf: dict, df_columns):
    # -------- Numerical Features --------
    numerical_features = []
    numerical_features += _filter_existing_columns(pre_conf.get("numerical", []), df_columns)
    numerical_features += _filter_existing_columns(pre_conf.get("time", []), df_columns)
    numerical_features += _filter_existing_columns(pre_conf.get("time_encoded", []), df_columns)

    # -------- Categorical Features --------
    categorical_conf = pre_conf.get("categorical", {})
    categorical_features = _filter_existing_columns(
        categorical_conf.get("regular", []), df_columns
    )
    binary_features = _filter_existing_columns(
        categorical_conf.get("binary", []), df_columns
    )

    
    # -------- Pipelines --------
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    bin_pipeline = Pipeline([
        ("encode", FunctionTransformer(encode_store_flag)),
        ("imputer", SimpleImputer(strategy="most_frequent"))
        
    ])

    transformers = []

    if 'cluster_pair_avg_duration' in df_columns:
        transformers.append((
            "avg_cluster_imputer",
            SimpleImputer(strategy="mean"),
            ["cluster_pair_avg_duration"]
        ))
    
    if numerical_features:
        transformers.append(("num", num_pipeline, numerical_features))

    if categorical_features:
        transformers.append(("cat", cat_pipeline, categorical_features))

    if binary_features:
        transformers.append(("bin", bin_pipeline, binary_features))

    # -------- Column Transformer --------
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough"  
    )

    preprocessor.set_output(transform="pandas")
    return preprocessor


def preprocess_data(df, target_col, is_train=False, use_log= True):
    df = data_cleaning(df) 

    if is_train:
        df = remove_outliers(df, target_col)  

    x, y = split_features_target(df, target_col)
    y = preprocess_target(y, use_log)
    
    return x, y


