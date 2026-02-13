from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator
from .Feature_Engineering import FeatureEngineering
from .Preprocessing import build_preprocessor

# class ModelPipeline:
#     def __init__(self, config, feature_engineering, numerical_features, categorical_features, polynomial_degree=1, alpha=0.01):
#         self.config = config
#         self.feature_engineering = feature_engineering
#         self.numerical_features = numerical_features
#         self.categorical_features = categorical_features
#         self.polynomial_degree = polynomial_degree
#         self.optimizer_alpha = alpha

#     def _build_preprocessor(self):

#         return build_preprocessor(config=self.config)

#     def _get_model(self, model_type: str):
#         if model_type == "ridge":
#             return Ridge(alpha=self.optimizer_alpha, fit_intercept=True, random_state=42)
#         elif model_type == "lasso":
#             return Lasso(alpha=self.optimizer_alpha, fit_intercept=True, max_iter=5000)
#         else:
#             raise ValueError(f"Unsupported model type: {model_type}")

#     def build_pipeline(self, model_type: str):
#         preprocessor = self._build_preprocessor()
#         model = self._get_model(model_type)

#         pipeline = Pipeline([
#             ("features", self.feature_engineering),
#             ("preprocess", preprocessor),
#             ("poly", PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)),
#             ("model", model)
#         ])

#         return pipeline
###################################################

# class ModelPipeline:
#     def __init__(self, config: dict, feature_engineering, build_preprocessor_fn):
#         self.config = config
#         self.feature_engineering = feature_engineering
#         self.build_preprocessor_fn = build_preprocessor_fn
#         # self.polynomial_degree = self.config.get("poly_degree", 1)

#     def _build_preprocessor(self, df_columns):
#         return self.build_preprocessor_fn(pre_conf=self.config["features"], df_columns=df_columns)


#     def _get_model(self, model_type: str):
#         alpha = self.config.get("optimizer", {}).get(model_type, 0.01)
#         if model_type == "ridge":
#             return Ridge(alpha=alpha, fit_intercept=True, random_state=42)
#         elif model_type == "lasso":
#             return Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
#         else:
#             raise ValueError(f"Unsupported model type: {model_type}")

#     def build_pipeline(self, model_type: str, df_columns):
#         preprocessor = self._build_preprocessor(df_columns)
#         model = self._get_model(model_type)

#         pipeline = Pipeline([
#             ("features", self.feature_engineering),
#             ("preprocess", preprocessor),
#             # ("poly", PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)),
#             ("model", model)
#         ])
#         return pipeline


#######################################

class ModelPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.feature_engineering = FeatureEngineering(config["features_engineering"])

    def _get_model(self, model_type: str):
        alpha = self.config["optimizer"].get(model_type, 0.01)
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
            df_columns=X_fe.columns,
            fe_config=self.config["features_engineering"]
        )
        

        model = self._get_model(model_type)
        self.pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        self.pipeline.fit(X_fe, y)
        return self

    def predict(self, X):
        X_fe = self.feature_engineering.transform(X)
        return self.pipeline.predict(X_fe)