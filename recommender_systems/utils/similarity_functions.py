import numpy as np

# Similarities


def cosine_similarity(u: np.ndarray, v: np.ndarray):
    """
    Computes de cosine similarity between two vectors
    Args:
        u: First vector
        v: Second vector

    Returns:
        The cosine similarity value
    """
    num = np.nansum(u * v)
    den = np.sqrt((np.nansum(u * u)) * (np.nansum(v * v)))
    return num / den


def pearson_correlation(u: np.ndarray, v: np.ndarray):
    """
    Computes the pearson correlation between two vectors
    Args:
        u: First vector
        v: Second vector

    Returns:
        The pearson correlation value
    """

    u_mean = np.nanmean(u)
    v_mean = np.nanmean(v)

    mean_deviation = (u - u_mean) * (v - v_mean)
    num = np.sum(mean_deviation)

    # We can get the common elements from mean_deviation
    common_indices = np.argwhere(~ np.isnan(mean_deviation))
    u_common = u[common_indices]
    v_common = v[common_indices]

    den = np.sqrt(np.sum(np.square(u_common - u_mean)) * np.sum(np.square(v_common - v_mean)))
    return num / den


def msd(u: np.ndarray, v: np.ndarray):
    """
    Computes the Mean Squared Difference between two vectors
    Args:
        u: The first vector
        v: The second vector

    Returns:
        The Mean Squared Difference value
    """
    difference = u - v
    common_indices = np.argwhere(~ np.isnan(difference))
    return len(common_indices) / np.nansum(np.square(difference))
