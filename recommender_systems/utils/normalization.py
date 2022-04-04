import numpy as np
import pandas as pd


class MeanCentering:
    """
    Mean centering normalization. Supports both users and items.
    """
    def __init__(self, user: bool = True):
        self.user = user
        self.means = []

    def transform(self, r: np.ndarray):
        n_u, n_i = r.shape
        if self.user:
            self.means = np.hstack([np.reshape(np.nanmean(r, axis=1), (-1, 1))]*n_i)
        else:
            self.means = np.vstack([np.reshape(np.nanmean(r, axis=0), (1, -1))]*n_u)
        return r - self.means

    def reverse_transform(self, r: np.ndarray):
        return r + self.means


class ZScoreNormalization:
    """
    Z Score normalization. Supports both users and items.
    """
    def __init__(self, user: bool = True):
        self.user = user
        self.means = []
        self.stds = []

    def transform(self, r: np.ndarray):
        n_u, n_i = r.shape
        if self.user:
            self.means = np.hstack([np.reshape(np.nanmean(r, axis=1), (-1, 1))] * n_i)
            self.stds = np.hstack([np.reshape(np.nanstd(r, axis=1), (-1, 1))] * n_i)
        else:
            self.means = np.vstack([np.reshape(np.nanmean(r, axis=0), (1, -1))] * n_u)
            self.stds = np.vstack([np.reshape(np.nanstd(r, axis=0), (1, -1))] * n_u)

        return (r - self.means) / self.stds

    def reverse_transform(self, r: np.ndarray):
        # We use hadamard product
        return np.multiply(self.stds, r) + self.means
