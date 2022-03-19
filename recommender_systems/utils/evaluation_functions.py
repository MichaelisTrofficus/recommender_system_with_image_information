import numpy as np


def mae(u: np.ndarray, v: np.ndarray):
    return np.mean(np.abs(u - v))


def rmse(u: np.ndarray, v: np.ndarray):
    return np.sqrt(np.mean(np.square(u - v)))
