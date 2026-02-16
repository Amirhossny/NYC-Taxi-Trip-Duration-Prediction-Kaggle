from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from .Feature_Engineering import FeatureEngineering
from sklearn.preprocessing import StandardScaler
from .Preprocessing import build_preprocessor

  
class ModelPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.feature_engineering = FeatureEngineering(config["features_engineering"])
        self.pipeline = None

    def _get_model(self, model_type: str):
        alpha = self.config.get("optimizer", {}).get(model_type, 0.01)
        if model_type == "ridge":
            return Ridge(alpha=alpha, random_state=42)
        elif model_type == "lasso":
            return Lasso(alpha=alpha, max_iter=5000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X, y, model_type: str):
        X_fe = self.feature_engineering.fit_transform(X)

        preprocessor = build_preprocessor(
            pre_conf=self.config["features"],
            df_columns=X_fe.columns
        )

        model = self._get_model(model_type)
        self.pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        self.pipeline.fit(X_fe, y)
        return self

    def predict(self, X):
        X_fe = self.feature_engineering.transform(X)
        return self.pipeline.predict(X_fe)