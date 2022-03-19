import numpy as np


def rating_average(neighbors_ratings: np.ndarray):
    """
    Computes a simple average between neighbors ratings
    Args:
        neighbors_ratings: A list of all the neighbors ratings

    Returns:
        The rating average as a prediction
    """
    return np.mean(neighbors_ratings)


def weighted_rating_average(neighbors_ratings: np.ndarray,
                            neighbors_weights: np.ndarray):
    """
    Computes the predicted rating as a weighted average. Weights are just simply a similarity
    measure.
    Args:
        neighbors_ratings: The neighbors ratings
        neighbors_weights: The neighbors similarities with the current user

    Returns:
        The weighted average as a prediction
    """
    return np.sum(neighbors_ratings * neighbors_weights) / np.sum(np.abs(neighbors_weights))
