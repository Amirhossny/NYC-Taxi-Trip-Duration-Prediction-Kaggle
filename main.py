import argparse
from pathlib import Path
import logging
from src.training import ModelPipeline
from src.Preprocessing import preprocess_data
from src.eval import evaluate
from src.logger import setup_logging, log_metrics
from src.Helper_Fun import load_data, load_config, save_model, load_model

# -------- Paths & Directories --------
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

# -------- Load & Preprocess Data --------
def load_and_preprocess_data():
    
    df_train = load_data(data_paths["train"])
    df_val = load_data(data_paths["val"])
    df_test = load_data(data_paths["test"])

    x_train, y_train = preprocess_data(df_train, target_col=target_col, is_train=True, use_log=use_log)
    x_val, y_val = preprocess_data(df_val, target_col=target_col, is_train=False, use_log=use_log)
    x_test, y_test = preprocess_data(df_test, target_col=target_col, is_train=False, use_log=use_log)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# -------- Train & Evaluate --------
def train_and_evaluate_model(model_name, x_train, y_train, x_val, y_val):
    logger.info(f"Training & validating {model_name} model")

    pipeline = ModelPipeline(config=config)
    pipeline.fit(x_train, y_train, model_type=model_name)

    save_model(pipeline, path=MODELS_PATH, name=f"{model_name}_model.pkl")

    metrics = evaluate(x_val, y_val, pipeline, name=model_name, use_log=use_log)
    log_metrics(logger, stage="val", model_name=model_name, metrics=metrics)

# -------- Test --------
def test_model(model_name, x_test, y_test):
    logger.info(f"Testing {model_name} model")

    pipeline = load_model(path=MODELS_PATH, name=f"{model_name}_model.pkl")
    metrics = evaluate(x_test, y_test, pipeline, name=model_name, use_log=use_log)
    log_metrics(logger, stage="test", model_name=model_name, metrics=metrics)

# -------- Argument Parser --------
def parse_args():
    parser = argparse.ArgumentParser(description="NYC Taxi Trip Duration - Training & Evaluation")
    parser.add_argument("--mode", choices=["train", "test", "all"], default="train", help="Run mode")
    parser.add_argument("--model", choices=["ridge", "lasso"], default="ridge", help="Model type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    if args.mode == "train":
        train_and_evaluate_model(args.model, x_train, y_train, x_val, y_val)
    elif args.mode == "test":
        test_model(args.model, x_test, y_test)
    elif args.mode == "all":
        train_and_evaluate_model(args.model, x_train, y_train, x_val, y_val)
        test_model(args.model, x_test, y_test)
    
    
    