import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from src.Feature_Engineering import build_features



# ---------------- Data Cleaning & Outlier Removal ----------------
def data_cleaning(df):
    
    df = df.dropna(subset=["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"])
    df = df[df["passenger_count"] > 0]
    return df

def remove_outliers(df, target_col, factor=6):
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return df[(df[target_col] >= lower) & (df[target_col] <= upper)]

# ---------------- Feature Engineering ----------------
def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["pickup_hour"] = dt.dt.hour
    df["pickup_weekday"] = dt.dt.weekday
    df["pickup_month"] = dt.dt.month
    df["is_weekend"] = df["pickup_weekday"].isin([5,6]).astype(int)
    df["rush_hour"] = ((df["pickup_hour"].between(7,9)) | (df["pickup_hour"].between(16,19))).astype(int)
    return df

def add_cyclical_encoding(df):
    df["pickup_hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["pickup_hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["pickup_weekday_sin"] = np.sin(2 * np.pi * df["pickup_weekday"] / 7)
    df["pickup_weekday_cos"] = np.cos(2 * np.pi * df["pickup_weekday"] / 7)
    df["pickup_month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12)
    df["pickup_month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12)
    # اختياري: ممكن تمسح الأعمدة الأصلية لو عايز
    df = df.drop(columns=["pickup_hour","pickup_weekday","pickup_month"], errors="ignore")
    return df

def add_distance_features(df):
    df["delta_latitude"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["delta_longitude"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["manhattan_distance"] = df["delta_latitude"].abs() + df["delta_longitude"].abs()
    # Haversine
    lat1, lon1, lat2, lon2 = map(np.radians, [df["pickup_latitude"], df["pickup_longitude"],
                                              df["dropoff_latitude"], df["dropoff_longitude"]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df["haversine_distance"] = 6371 * c
    return df

def encode_store_flag(df):
    df = df.copy()
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"Y":1,"N":0})
    return df

# ---------------- Target Preprocessing ----------------
def preprocess_target(y, use_log=True):
    if use_log:
        y = np.log1p(y)
    return y

# ---------------- Main ----------------
if __name__ == "__main__":
    # Load data
    train_path = "Data/train.csv"
    val_path = "Data/val.csv"
    target_col = "trip_duration"

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Clean + outliers
    df_train = data_cleaning(df_train)
    df_train = remove_outliers(df_train, target_col)
    # df_train = encode_store_flag(df_train)
   
    
    # Features
    # print("BEFORE FE:", df_train.columns.tolist())
    # df_train = add_time_features(df_train)
    # df_train = add_cyclical_encoding(df_train)
    # df_train = add_distance_features(df_train)
    # print("AFTER FE:", df_train.columns.tolist())

    # Split features/target
    X_train = df_train.drop(columns=[target_col, "id"], errors="ignore") #
    y_train = preprocess_target(df_train[target_col])

    df_val = data_cleaning(df_val)
    df_val = remove_outliers(df_val, target_col)
    # df_val = encode_store_flag(df_val)

    # df_val = add_time_features(df_val)
    # df_val = add_cyclical_encoding(df_val)
    # df_val = add_distance_features(df_val)

    X_val = df_val.drop(columns=[target_col, "id"], errors="ignore")
    y_val = preprocess_target(df_val[target_col])

    # print("FINAL COLUMNS:", X_train.columns.tolist())

    

    X_train = build_features(X_train, use_time_features=True, use_distance_features=True, use_manhattan=True, use_haversine=True, use_bearing=False)

    X_val = build_features(X_val,use_time_features=True,use_distance_features=True,use_manhattan=True,use_haversine=True,use_bearing=False)

    print("FINAL COLUMNS:", X_train.columns.tolist())
    print("FINAL COLUMNS:", X_val.columns.tolist())
    
    # Fit Ridge model
    model = Ridge(alpha=0.01)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model.fit(X_train_scaled, y_train)

    # Predict & Evaluate
    y_pred = model.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"Ridge RMSE = {rmse:.4f} & R2 = {r2:.4f}")
