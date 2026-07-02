import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def binary_cross_entropy(y_true, y_pred):
    # clipping for avoid log0
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return np.mean(
        -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    )


def binary_cross_entropy_prime(y_true, y_pred):
    # clipping for avoid division by zero
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)