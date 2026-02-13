
from pathlib import Path
import logging
from src.training import ModelPipeline
from src.Preprocessing import preprocess_data
from src.eval import evaluate
from src.logger import setup_logging, log_metrics
from src.Helper_Fun import load_data, load_config, save_model, load_model

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
    
    
    train_and_evaluate_model("ridge", x_train, y_train, x_val, y_val, use_log)
    
    
    # for model_name in ["ridge", "lasso"]:
    #     train_and_evaluate_model(model_name, x_train, y_train, x_val, y_val, use_log)
    #     test_model(model_name, x_test, y_test, use_log)
        
    
    
    # from src.Feature_Engineering import FeatureEngineering
    
    # print("BEFORE FE:", x_train.columns.tolist())
    # feature_engineering = FeatureEngineering(fe_con)
    # print("FINAL FEATURES COUNT:", x_train.shape)
    # print(x_train.dtypes)
    # X_fe = feature_engineering.fit_transform(x_train)
    # print("AFTER FE:", X_fe.columns.tolist())
    # # train_and_evaluate_model("ridge", x_train, y_train, x_val, y_val, use_log)
    # feature_engineering = FeatureEngineering(fe_con)
    # X_fe = feature_engineering.fit_transform(x_train)
    # print(X_fe.columns)
    
    # preprocessor = build_preprocessor(config)
    # X_transformed = preprocessor.fit_transform(x_train)
    # print(X_transformed.shape)
    # print(X_transformed.columns.tolist())
    
    # # print(len(x_train), len(y_train))
    # # print(y_train.describe())
    
    # model = LinearRegression()
    # model.fit(x_train.select_dtypes(include=np.number), y_train)