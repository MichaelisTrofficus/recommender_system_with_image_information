import numpy as np
from typing import Callable
from tqdm import tqdm

from utils.global_utils import generate_ratings_matrix
from utils.similarity_functions import cosine_similarity, pearson_correlation, msd
from utils.rating_functions import rating_average, weighted_rating_average
from utils.evaluation_functions import mae, rmse

import pandas as pd


class KNN:
    def __init__(self, data: np.ndarray, k: int, sim_fn: str, rating_fn: str):
        """
        A Class that implements the
        Args:
            data: The training data
            k: The number of neighbors
            sim_fn: Similarity method name
            rating_fn: Rating method name
        """
        self.data = generate_ratings_matrix(data)
        self.k = k
        self.sim_fn = self._get_sim_fn(sim_fn)
        self.rating_fn = self._get_rating_fn(rating_fn)

    def _get_sim_fn(self, sim_fn: str):
        """
        Gets the similarity Callable method
        Args:
            sim_fn: The similarity method name
        """
        if sim_fn == "cosine":
            self.sim_fn = cosine_similarity
        elif sim_fn == "pearson":
            self.sim_fn = pearson_correlation
        elif sim_fn == "msd":
            self.sim_fn = msd
        else:
            raise ValueError("Similarity functions must be one of the following:"
                             " `cosine`, `pearson`, `msd`")

    def _get_rating_fn(self, rating_fn: str):
        """
        Gets the rating Callable method
        Args:
            rating_fn: The rating method name
        """
        if rating_fn == "average":
            self.rating_fn = rating_average
        elif rating_fn == "w_average":
            self.rating_fn = weighted_rating_average
        else:
            raise ValueError(
                "Rating functions must be one of the following: "
                "`average`, `w_average`"
            )

    def _compute_sim(self):
        sim_matrix = np.zeros((943, 943))

        for i, u in tqdm(enumerate(self.data)):
            for j, v in enumerate(self.data):
                sim_matrix[i][j] = pearson_correlation(u, v)
        return sim_matrix

    def evaluate(self, test: np.ndarray):
        print("Computing similarity matrix ...")
        sim_matrix = self._compute_sim()

        print("Making predictions ...")
        predicted_ratings = []
        true_ratings = test[:, 2]
        for i in test:
            user_id = i[0] - 1
            item_id = i[1] - 1

            most_similar_neighbors = np.argsort(sim_matrix[user_id])[::-1][1:]
            neighbors_ratings = []

            for neighbor in most_similar_neighbors:
                if len(neighbors_ratings) > self.k:
                    break

                neighbor_rating = self.data[neighbor][item_id]
                if not np.isnan(neighbor_rating):
                    neighbors_ratings.append(neighbor_rating)

            predicted_ratings.append(np.mean(neighbors_ratings))

        # Convert it into a numpy array
        predicted_ratings = np.asarray(predicted_ratings)

        return {
            "mae": mae(predicted_ratings, true_ratings),
            "rmse": rmse(predicted_ratings, true_ratings)
        }


train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
KNN(data=train.values, k=30, rating_fn=None, sim_fn=None).evaluate(test.values)




