import numpy as np


def determinant(m: np.array, j: int = 0) -> int:
    """
    Implements Laplace Expansion in a recursive way

    Args:
        m (np.array): An input matrix
        j (int): The column index where the algorithm will be applied
    Returns:
        The determinant of the matrix provided
    """

    n = m.shape[0]
    if n == 2:
        return m[0][0]*m[1][1] - m[1][0]*m[0][1]

    det = 0
    for k in range(n):
        det += ((-1) ** (k + j))*m[k][j]*determinant(np.delete(np.delete(m, k, axis=0), j, axis=1))
    return det
