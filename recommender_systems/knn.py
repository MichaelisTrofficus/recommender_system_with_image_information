from typing import List

import pandas as pd
from tqdm import tqdm

import numpy as np

from recommender_systems.utils.similarity_functions import pearson_correlation
from recommender_systems.utils.evaluation_functions import mae, rmse


class KNNUserUser:
    def __init__(self,
                 max_neighbors: int,
                 significance_weighting: int = 1):
        """
        A Class that implements a Nearest Neighbors algorithm for collaborative filtering
        Args:
            max_neighbors: The maximum number of neighbors
            significance_weighting:
        """
        self.max_neighbors = max_neighbors
        self.significance_weighting = significance_weighting
        self.sim_matrix = None

    @staticmethod
    def calculate_rating(neighbors_ratings: List[float], neighbors_similarities: List[float]):
        neighbors_ratings = np.asarray(neighbors_ratings)
        neighbors_similarities = np.asarray(neighbors_similarities)
        return np.nansum(np.multiply(neighbors_ratings, neighbors_similarities)) /\
               np.nansum(np.abs(neighbors_similarities))

    @staticmethod
    def compute_sim_matrix(r: np.ndarray):
        sim_matrix = np.zeros((943, 943))

        for i, u in tqdm(enumerate(r)):
            for j, v in enumerate(r):
                sim_matrix[i][j] = pearson_correlation(u, v)
        return sim_matrix

    def fit(self, r: np.ndarray, save_checkpoint: bool = True):
        self.sim_matrix = self.compute_sim_matrix(r)

        if save_checkpoint:
            with open("./sim_matrix_user_20_zscore_no_significance.npy", "wb") as f:
                np.save(f, self.sim_matrix)

    def evaluate(self, r: np.ndarray, test: pd.DataFrame, load_checkpoint_path: str = ""):

        test = test.copy()
        test_arr = test.values

        if load_checkpoint_path:
            with open(load_checkpoint_path, "rb") as f:
                self.sim_matrix = np.load(f)

        print("Making predictions ...")
        predicted_ratings = []

        for i in tqdm(test_arr):
            user_id = int(i[0] - 1)
            item_id = int(i[1] - 1)

            most_similar_neighbors = np.argsort(self.sim_matrix[user_id])[::-1]
            neighbors_ratings = []
            neighbors_similarities = []

            for neighbor_id in most_similar_neighbors:
                if neighbor_id == user_id:
                    continue

                if len(neighbors_ratings) > self.max_neighbors:
                    break

                neighbor_rating = r[neighbor_id][item_id]
                neighbor_similarity = self.sim_matrix[user_id][neighbor_id]

                if not np.isnan(neighbor_rating):
                    neighbors_ratings.append(neighbor_rating)
                    neighbors_similarities.append(neighbor_similarity)

            predicted_ratings.append(self.calculate_rating(neighbors_ratings, neighbors_similarities))

        # Convert it into a numpy array
        test["rating"] = np.asarray(predicted_ratings)
        return test
