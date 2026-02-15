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
# def encode_store_flag(df):
#     df = df.copy()
#     df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"Y":1,"N":0})
#     return df[['store_and_fwd_flag']]

# --- build preprocessor pipeline ---



# def build_preprocessor(pre_conf, df_columns, fe_config):
#     numerical_features = [f for f in pre_conf.get("numerical", []) if f in df_columns]

#     time_encoded_features = [f for f in pre_conf.get("time_encoded", []) if f in df_columns]
#     numerical_features += time_encoded_features 

#     categorical = pre_conf.get("categorical", {})
#     categorical_features = [f for f in categorical.get("regular", []) if f in df_columns]
#     binary_features = [f for f in categorical.get("binary", []) if f in df_columns]

#     num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
#     cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
#                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
#     bin_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
#                              ("encode", FunctionTransformer(encode_store_flag))])

#     transformers = [
#         ("num", num_pipeline, numerical_features),
#         ("cat", cat_pipeline, categorical_features),
#         ("bin", bin_pipeline, binary_features)
#     ]

#     preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
#     preprocessor.set_output(transform="pandas")
#     return preprocessor

# def build_preprocessor(self, df_columns):

#     numerical_features = [
#         "passenger_count",
#         "pickup_longitude", "pickup_latitude",
#         "dropoff_longitude", "dropoff_latitude",
#         "pickup_hour", "pickup_day", "pickup_month", "pickup_weekday",
#         "is_weekend", "rush_hour",
#         "delta_latitude", "delta_longitude",
#         "manhattan_distance", "haversine_distance", "bearing"
#     ]

#     categorical_features = [
#         "vendor_id",
#         "store_and_fwd_flag",
#         "pickup_cluster",
#         "dropoff_cluster"
#     ]

#     return ColumnTransformer(
#         transformers=[
#             ("num", StandardScaler(), numerical_features),
#             ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
#         ],
#         remainder="drop"
#     )

def preprocess_data(df, target_col, is_train=False, use_log= True):
    df = data_cleaning(df) 

    if is_train:
        df = remove_outliers(df, target_col)  

    x, y = split_features_target(df, target_col)
    y = preprocess_target(y, use_log)
    
    return x, y


# def build_preprocessor(config):
#     numerical_features = (
#         config["features"]["numerical"]["base"] +
#         config["features"]["numerical"]["distance"] +
#         config["features"]["numerical"]["cluster"]
#     )
#     categorical_features = config["features"]["categorical"]["regular"]
#     binary_features = config["features"]["categorical"]["binary"]

#     # numerical pipeline
#     num_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="mean")),
#         ("scaler", StandardScaler())
#     ])

#     # categorical pipeline
#     cat_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
#     ])

#     # binary pipeline
#     bin_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("mapper", FunctionTransformer(encode_store_flag))
#     ])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", num_pipeline, numerical_features),
#             ("cat", cat_pipeline, categorical_features),
#             ("bin", bin_pipeline, binary_features)
#         ],
#         remainder="drop"  # كل الأعمدة الغير مستخدمة هتتشال
#     )

#     preprocessor.set_output(transform="pandas")
#     return preprocessor