import joblib
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.Helper_Fun import load_data
from src.Preprocessing import data_cleaning
from src.Preprocessing import remove_ouliers

def save_cluster_model(kmeans, cluster_pair_avg, path="Kmeans_Helper"):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)  
    joblib.dump(kmeans, path / "kmeans.pkl")
    joblib.dump(cluster_pair_avg, path / "cluster_pair_avg.pkl")

def load_cluster_model(path="Kmeans_Helper"):
    path = Path(path)
    kmeans = joblib.load(path / "kmeans.pkl")
    cluster_pair_avg = joblib.load(path / "cluster_pair_avg.pkl")
    return kmeans, cluster_pair_avg


def train_and_save_cluster(df_train, n_clusters=100, cluster_path="cluster/"):
   
    cluster_path = Path(cluster_path)
    cluster_path.mkdir(parents=True, exist_ok=True)
    
    coords = np.vstack((
        df_train[['pickup_latitude', 'pickup_longitude']].values,
        df_train[['dropoff_latitude', 'dropoff_longitude']].values,
        ))
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,random_state=42,batch_size=1000)
    kmeans.fit(coords)
    
    df_train['pickup_cluster'] = kmeans.predict(df_train[['pickup_latitude','pickup_longitude']].values)
    df_train['dropoff_cluster'] = kmeans.predict(df_train[['dropoff_latitude','dropoff_longitude']].values)
    
    cluster_pair_avg = df_train.groupby(['pickup_cluster','dropoff_cluster'])['trip_duration'].mean().to_dict()
    
    save_cluster_model(kmeans, cluster_pair_avg, path="Kmeans_Helper")
    
    print(f"Saved KMeans and cluster_pair_avg to {cluster_path}")
    
    return kmeans, cluster_pair_avg

if __name__ == "__main__":
    
    df_train = load_data("data/train.csv")
    df_train= data_cleaning(df_train)
    df_train = remove_ouliers(df_train)
    
    train_and_save_cluster(df_train, n_clusters=100, cluster_path="Kmeans_Helper/")
    
    