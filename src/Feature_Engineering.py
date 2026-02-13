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


'''new version with clustering - commented out for now'''
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, config: dict):
        self.config = config
        self.kmeans = None
        self.cluster_pair_avg = None

    def fit(self, X, y=None):
        if self.config.get("use_cluster_features", False):
            clustering = self.config.get("clustering", None)
            if clustering is None:
                raise ValueError("Clustering config is required when use_cluster_features=True")

            self.kmeans = joblib.load(clustering["cluster_model"])
            self.cluster_pair_avg = joblib.load(clustering["cluster_pair_avg"])
        return self

    def transform(self, X):
        df = X.copy()

        if self.config.get("use_time_features", False):
            df = self._add_time_features(df)

        if self.config.get("use_distance_features", False):
            df = self._add_distance_features(df)

        if self.config.get("use_cluster_features", False):
            pickup = self.kmeans.predict(df[['pickup_latitude','pickup_longitude']].values)
            dropoff = self.kmeans.predict(df[['dropoff_latitude','dropoff_longitude']].values)

            df['pickup_cluster'] = pickup
            df['dropoff_cluster'] = dropoff

            df['cluster_pair_avg_duration'] = [
                self.cluster_pair_avg.get((p, d), np.nan)
                for p, d in zip(pickup, dropoff)
            ]

        return df

    # -------------------
    # Time Features
    # -------------------
    def _add_time_features(self, df):
        if "pickup_datetime" not in df.columns:
            return df
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_day'] = df['pickup_datetime'].dt.day
        df['pickup_month'] = df['pickup_datetime'].dt.month
        df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
        df['is_weekend'] = df['pickup_weekday'].isin([5, 6]).astype(int)
        df['rush_hour'] = ((df['pickup_hour'].between(7, 9)) |
                           (df['pickup_hour'].between(16, 19))).astype(int)
        df = df.drop(columns=["pickup_datetime"], errors="ignore")
        return df

    # -------------------
    # Distance Features
    # -------------------
    def _add_distance_features(self, df):
        df['delta_latitude'] = df['dropoff_latitude'] - df['pickup_latitude']
        df['delta_longitude'] = df['dropoff_longitude'] - df['pickup_longitude']

        if self.config.get("use_manhattan", False):
            df['manhattan_distance'] = df['delta_latitude'].abs() + df['delta_longitude'].abs()
        if self.config.get("use_haversine", False):
            df['haversine_distance'] = self._haversine(
                df['pickup_latitude'], df['pickup_longitude'],
                df['dropoff_latitude'], df['dropoff_longitude']
            )
        if self.config.get("use_bearing", False):
            df['bearing'] = self._bearing(
                df['pickup_latitude'], df['pickup_longitude'],
                df['dropoff_latitude'], df['dropoff_longitude']
            )
        return df

    # -------------------
    # Utility Functions
    # -------------------
    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c

    @staticmethod
    def _bearing(lat1, lon1, lat2, lon2):
        dlon = np.radians(lon2 - lon1)
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        return (np.degrees(np.arctan2(x, y)) + 360) % 360


    
###########################################
'''old version'''
# class FeatureEngineering(BaseEstimator, TransformerMixin):
#     def __init__(self, config: FeatureConfig):
#         self.config = config
#         self.kmeans = None
#         self.cluster_pair_avg = None
          


#     def fit(self, X, y=None):
#         # load cluster models ONCE during fit
#         if self.config.use_cluster_features:
#             if self.config.clustering is None:
#                 raise ValueError("Clustering config is required when use_cluster_features=True")

#             self.kmeans = joblib.load(self.config.clustering.cluster_model)
#             self.cluster_pair_avg = joblib.load(self.config.clustering.cluster_pair_avg)

#         return self

#     def transform(self, X):
#         df = X.copy()

#         if self.config.use_time_features:
#             df = self._add_time_features(df)

#         if self.config.use_distance_features: 
#             df = self._add_distance_features(df)

#         if self.config.use_cluster_features:
#             pickup = self.kmeans.predict(df[['pickup_latitude','pickup_longitude']].values)
#             dropoff = self.kmeans.predict(df[['dropoff_latitude','dropoff_longitude']].values)

#             df['pickup_cluster'] = pickup
#             df['dropoff_cluster'] = dropoff

#             df['cluster_pair_avg_duration'] = [
#                 self.cluster_pair_avg.get((p, d), np.nan)
#                 for p, d in zip(pickup, dropoff)]

#         return df
#     # -------------------
#     # Time Features
#     # -------------------
#     def _add_time_features(self, df):
#         if "pickup_datetime" not in df.columns:
#             return df
#         df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
#         df['pickup_hour'] = df['pickup_datetime'].dt.hour
#         df['pickup_day'] = df['pickup_datetime'].dt.day
#         df['pickup_month'] = df['pickup_datetime'].dt.month
#         df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
#         df['is_weekend'] = df['pickup_weekday'].isin([5, 6]).astype(int)
#         df['rush_hour'] = ((df['pickup_hour'].between(7, 9)) |
#                            (df['pickup_hour'].between(16, 19))).astype(int)
#         df = df.drop(columns=["pickup_datetime"], errors="ignore")
#         return df

#     # -------------------
#     # Distance Features
#     # -------------------
#     def _add_distance_features(self, df):
#         df['delta_latitude'] = df['dropoff_latitude'] - df['pickup_latitude']
#         df['delta_longitude'] = df['dropoff_longitude'] - df['pickup_longitude']

#         if self.config.use_manhattan:
#             df['manhattan_distance'] = df['delta_latitude'].abs() + df['delta_longitude'].abs()
#         if self.config.use_haversine:
#             df['haversine_distance'] = self._haversine(
#                 df['pickup_latitude'], df['pickup_longitude'],
#                 df['dropoff_latitude'], df['dropoff_longitude']
#             )
#         if self.config.use_bearing:
#             df['bearing'] = self._bearing(
#                 df['pickup_latitude'], df['pickup_longitude'],
#                 df['dropoff_latitude'], df['dropoff_longitude']
#             )
#         return df

#     # -------------------
#     # Utility Functions
#     # -------------------
#     @staticmethod
#     def _haversine(lat1, lon1, lat2, lon2):
#         lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
#         dlat = lat2 - lat1
#         dlon = lon2 - lon1
#         a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
#         c = 2 * np.arcsin(np.sqrt(a))
#         return 6371 * c

#     @staticmethod
#     def _bearing(lat1, lon1, lat2, lon2):
#         dlon = np.radians(lon2 - lon1)
#         lat1 = np.radians(lat1)
#         lat2 = np.radians(lat2)
#         x = np.sin(dlon) * np.cos(lat2)
#         y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
#         return (np.degrees(np.arctan2(x, y)) + 360) % 360


