import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def evaluate(x, y, model, name, use_log=True):
    preds = model.predict(x)

    if use_log:
        r2 = r2_score(y, preds)
        rmse = np.sqrt(
            mean_squared_error(
                np.expm1(y),
                np.expm1(preds)
            )
        )
        return {
            "r2_log": r2,
            "rmse": rmse
        }
    else:
        r2 = r2_score(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        return {
            "r2": r2,
            "rmse": rmse
        }