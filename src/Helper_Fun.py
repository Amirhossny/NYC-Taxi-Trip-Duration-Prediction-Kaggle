import pandas as pd
import yaml
import joblib
import os

def load_data(path):
    df = pd.read_csv(path)
    return df

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def save_model(model, path="Models/", name="model.pkl"):
    joblib.dump(model, os.path.join(path, name))


def load_model(path="Models/", name="model.pkl"):
    return joblib.load(os.path.join(path, name)) 

