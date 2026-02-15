import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

#note to write in cv
'''
Engineered spatial features using MiniBatchKMeans clustering on pickup and dropoff
coordinates, and derived cluster-pair target encoding to capture geographic trip
duration patterns, implemented as a fully sklearn-compatible transformer
'''

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config
        self.use_time_features = config.get("use_time_features", True)
        self.use_distance_features = config.get("use_distance_features", True)
        self.use_haversine = config.get("use_haversine", False)
        self.use_manhattan = config.get("use_manhattan", False)
        self.use_bearing = config.get("use_bearing", False)
        self.use_cluster_features = config.get("use_cluster_features", False)

        self.kmeans = None
        self.cluster_pair_avg = None

    def fit(self, X, y=None):
        if self.use_cluster_features:
            clustering = self.config["clustering"]
            self.kmeans = joblib.load(clustering["cluster_model"])
            self.cluster_pair_avg = joblib.load(clustering["cluster_pair_avg"])
        return self

    def transform(self, X):
        df = X.copy()

        df = self._encode_store_and_fwd_flag(df)
        
        if self.use_time_features:
            df = self._add_time_features(df)
            df = self._add_cyclical_encoding(df)

        if self.use_distance_features:
            df = self._add_distance_features(df)

        if self.use_cluster_features:
            df = self._add_cluster_features(df)

        return df
    # ---------- Binary Encoding ----------
    def _encode_store_and_fwd_flag(self, df):
        if "store_and_fwd_flag" in df.columns:
            df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"Y": 1, "N": 0})
        return df
    # ---------- Time Features ----------
    def _add_time_features(self, df):
        if "pickup_datetime" not in df.columns:
            raise ValueError("pickup_datetime column is required")

        dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df["pickup_hour"] = dt.dt.hour
        df["pickup_weekday"] = dt.dt.weekday
        df["pickup_month"] = dt.dt.month
        df["is_weekend"] = df["pickup_weekday"].isin([5, 6]).astype(int)
        df["rush_hour"] = (
            df["pickup_hour"].between(7, 9) |
            df["pickup_hour"].between(16, 19)
        ).astype(int)

        df = df.drop(columns=["pickup_datetime"])
        return df

    def _add_cyclical_encoding(self, df):
        df["pickup_hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
        df["pickup_hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
        df["pickup_weekday_sin"] = np.sin(2 * np.pi * df["pickup_weekday"] / 7)
        df["pickup_weekday_cos"] = np.cos(2 * np.pi * df["pickup_weekday"] / 7)
        df["pickup_month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12)
        df["pickup_month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12)
        return df

    # ---------- Distance Features ----------
    def _add_distance_features(self, df):
        df["delta_latitude"] = df["dropoff_latitude"] - df["pickup_latitude"]
        df["delta_longitude"] = df["dropoff_longitude"] - df["pickup_longitude"]

        if self.use_manhattan:
            df["manhattan_distance"] = (
                df["delta_latitude"].abs() + df["delta_longitude"].abs()
            )

        if self.use_haversine:
            df["haversine_distance"] = self._haversine(
                df["pickup_latitude"], df["pickup_longitude"],
                df["dropoff_latitude"], df["dropoff_longitude"]
            )

        if self.use_bearing:
            df["bearing"] = self._bearing(
                df["pickup_latitude"], df["pickup_longitude"],
                df["dropoff_latitude"], df["dropoff_longitude"]
            )

        return df

    # ---------- Cluster Features ----------
    def _add_cluster_features(self, df):
        if self.kmeans is None or self.cluster_pair_avg is None:
            raise RuntimeError("FeatureEngineering must be fitted before transform")

        # Predict clusters
        pickup = self.kmeans.predict(df[["pickup_latitude", "pickup_longitude"]].values)
        dropoff = self.kmeans.predict(df[["dropoff_latitude", "dropoff_longitude"]].values)

        df["pickup_cluster"] = pickup
        df["dropoff_cluster"] = dropoff

        global_avg = np.mean(list(self.cluster_pair_avg.values()))

        df["cluster_pair_avg_duration"] = [
            self.cluster_pair_avg.get((p, d), global_avg) ##
            for p, d in zip(pickup, dropoff)
        ]

        return df

    # ---------- Utilities ----------
    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2) ** 2 +
            np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c

    @staticmethod
    def _bearing(lat1, lon1, lat2, lon2):
        dlon = np.radians(lon2 - lon1)
        lat1, lat2 = map(np.radians, [lat1, lat2])
        x = np.sin(dlon) * np.cos(lat2)
        y = (
            np.cos(lat1) * np.sin(lat2)
            - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        )
        return (np.degrees(np.arctan2(x, y)) + 360) % 360


# # ===============================
# # Time Features
# # ===============================
# def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()

#     if "pickup_datetime" not in df.columns:
#         raise ValueError("pickup_datetime column is required")

#     dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")

#     df["pickup_hour"] = dt.dt.hour
#     df["pickup_weekday"] = dt.dt.weekday
#     df["pickup_month"] = dt.dt.month

#     df["is_weekend"] = df["pickup_weekday"].isin([5, 6]).astype(int)

#     df["rush_hour"] = (
#         df["pickup_hour"].between(7, 9)
#         | df["pickup_hour"].between(16, 19)
#     ).astype(int)

#     df = df.drop(columns=["pickup_datetime"])
#     return df


# def add_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()

#     df["pickup_hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
#     df["pickup_hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)

#     df["pickup_weekday_sin"] = np.sin(2 * np.pi * df["pickup_weekday"] / 7)
#     df["pickup_weekday_cos"] = np.cos(2 * np.pi * df["pickup_weekday"] / 7)

#     df["pickup_month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12)
#     df["pickup_month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12)

#     return df


# # ===============================
# # Distance Features
# # ===============================
# def add_distance_features(
#     df: pd.DataFrame,
#     use_manhattan: bool = True,
#     use_haversine: bool = True,
#     use_bearing: bool = False
# ) -> pd.DataFrame:

#     df = df.copy()

#     df["delta_latitude"] = df["dropoff_latitude"] - df["pickup_latitude"]
#     df["delta_longitude"] = df["dropoff_longitude"] - df["pickup_longitude"]

#     if use_manhattan:
#         df["manhattan_distance"] = (
#             df["delta_latitude"].abs() + df["delta_longitude"].abs()
#         )

#     if use_haversine:
#         df["haversine_distance"] = haversine(
#             df["pickup_latitude"],
#             df["pickup_longitude"],
#             df["dropoff_latitude"],
#             df["dropoff_longitude"]
#         )

#     if use_bearing:
#         df["bearing"] = bearing(
#             df["pickup_latitude"],
#             df["pickup_longitude"],
#             df["dropoff_latitude"],
#             df["dropoff_longitude"]
#         )

#     return df


# # ===============================
# # Encoders
# # ===============================
# def encode_store_and_fwd_flag(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"Y": 1, "N": 0})
#     return df


# # ===============================
# # Main Feature Engineering
# # ===============================
# def build_features(
#     df: pd.DataFrame,
#     use_time_features: bool = True,
#     use_distance_features: bool = True,
#     use_manhattan: bool = True,
#     use_haversine: bool = True,
#     use_bearing: bool = False
# ) -> pd.DataFrame:

#     df = df.copy()

#     if use_time_features:
#         df = add_time_features(df)
#         df = add_cyclical_encoding(df)

#     if use_distance_features:
#         df = add_distance_features(
#             df,
#             use_manhattan=use_manhattan,
#             use_haversine=use_haversine,
#             use_bearing=use_bearing
#         )

#     df = encode_store_and_fwd_flag(df)

#     return df


# # ===============================
# # Utilities
# # ===============================
# def haversine(lat1, lon1, lat2, lon2):
#     lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1

#     a = (
#         np.sin(dlat / 2) ** 2
#         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
#     )

#     c = 2 * np.arcsin(np.sqrt(a))
#     return 6371 * c


# def bearing(lat1, lon1, lat2, lon2):
#     lat1, lat2 = map(np.radians, [lat1, lat2])
#     dlon = np.radians(lon2 - lon1)

#     x = np.sin(dlon) * np.cos(lat2)
#     y = (
#         np.cos(lat1) * np.sin(lat2)
#         - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
#     )

#     return (np.degrees(np.arctan2(x, y)) + 360) % 360