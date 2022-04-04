import numpy as np
from sklearn.model_selection import train_test_split


def split(data: np.ndarray, test_size: float = 0.1):
    return train_test_split(data, test_size=test_size, random_state=192, shuffle=True)


def generate_ratings_matrix(data_rows: np.ndarray):
    r = np.zeros((943, 1682))
    r[:] = np.nan
    for i in data_rows:
        r[int(i[0]) - 1][int(i[1]) - 1] = float(i[2])
    return r
