import numpy as np


def mae(u: np.ndarray, v: np.ndarray):
    return np.nanmean(np.abs(u - v))


def rmse(u: np.ndarray, v: np.ndarray):
    return np.sqrt(np.nanmean(np.square(u - v)))
