
from pathlib import Path
import logging
from src.training import ModelPipeline
from src.Preprocessing import preprocess_data
from src.eval import evaluate
from src.logger import setup_logging, log_metrics
from src.Helper_Fun import load_data, load_config, save_model, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from src.Feature_Engineering import FeatureEngineering
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR / "Models"
MODELS_PATH.mkdir(exist_ok=True)
CONFIG_PATH = BASE_DIR / "config.yaml"



# -------- Logging --------
setup_logging("nyc_taxi.log")
logger = logging.getLogger(__name__)


# -------- Config --------
config = load_config(CONFIG_PATH)

data_paths = config["data"]["paths"]
target_col = config["data"]["target_col"]
use_log = config["log_target"]
fe_con = config["features_engineering"]
pre_con = config["features"]
poly_degree = config["poly_degree"]



# -------- Load Data --------
df_train = load_data(data_paths["train"])
df_val   = load_data(data_paths["val"])
df_test  = load_data(data_paths["test"])

x_train,y_train = preprocess_data(df_train, target_col=target_col, is_train=True, use_log=use_log)
x_val, y_val = preprocess_data(df_val, target_col=target_col, is_train=False, use_log=use_log)
x_test, y_test = preprocess_data(df_test, target_col=target_col, is_train=False, use_log=use_log)





def train_and_evaluate_model(model_name, x_train, y_train, x_val, y_val, use_log):

    logger.info(f"Training & validating {model_name} model")


    pipeline = ModelPipeline(config=config)
    pipeline.fit(x_train, y_train, model_type=model_name)
    save_model(pipeline, path=MODELS_PATH, name=f"{model_name}_model.pkl")
    metrics = evaluate(x_val, y_val, pipeline, name=model_name, use_log=use_log)
    log_metrics(logger, stage="val", model_name=model_name, metrics=metrics)


# -------- Test --------
def test_model(model_name, x_test, y_test, use_log):

    logger.info(f"Testing {model_name} model")

    model = load_model( path=MODELS_PATH, name=f"{model_name}_model.pkl")

    metrics = evaluate(x_test, y_test, model, name=model_name, use_log=use_log)

    log_metrics(logger, stage="test", model_name=model_name, metrics=metrics)
        




        
      


if __name__ == "__main__":
    
    
    # train_and_evaluate_model("ridge", x_train, y_train, x_val, y_val, use_log)
    
    
    fe = FeatureEngineering(fe_con)
    x_train = fe.fit_transform(x_train)
    x_val = fe.transform(x_val)

    scaler = StandardScaler()

    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=x_train.columns,
        index=x_train.index
    )

    x_val_scaled = pd.DataFrame(
        scaler.transform(x_val),
        columns=x_val.columns,
        index=x_val.index
    )

    
    # print("NaNs in val:")
    # print(
    #     x_val.isna()
    #     .sum()
    #     .sort_values(ascending=False)
    #     .head(10)
    # )

    # print(
    #     "cluster_pair_avg_duration NaN ratio:",
    #     x_val["cluster_pair_avg_duration"].isna().mean()
    # )

    # print("Train columns:", x_train.columns.tolist())
    # print("Val columns:", x_val.columns.tolist())

    # print("NaNs in train:")
    # print(x_train.isna().sum().sort_values(ascending=False).head(10))
    
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    preds = model.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)

    print(f"Ridge RMSE = {rmse:.4f} & R2 = {r2:.4f}")