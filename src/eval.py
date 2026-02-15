import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def inverse_target(y_transformed, use_log=True):
    if use_log:
        return np.expm1(y_transformed)
    return y_transformed

def evaluate(X_test, y_test, model, name, use_log=True):
    pred_trans = model.predict(X_test)
    y_pred_real = inverse_target(pred_trans, use_log)
    y_true_real = inverse_target(y_test, use_log)

    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    r2 = r2_score(y_true_real, y_pred_real)
    # print(f"{name} RMSE = {rmse:.4f} & R2 = {r2:.4f}")
    return {'r2': r2, 'rmse': rmse}