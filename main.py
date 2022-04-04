import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from recommender_systems.utils.global_utils import generate_ratings_matrix
from recommender_systems.utils.normalization import ZScoreNormalization
from recommender_systems.knn import KNNUserUser

train = pd.read_csv("./data/train_10_ratings.csv")
test = pd.read_csv("./data/test_10_ratings.csv")

r = generate_ratings_matrix(train.values)
z_score_norm = ZScoreNormalization()
r_centered = z_score_norm.transform(r)

# knn_user = KNNUserUser(max_neighbors=20).fit(r_centered, save_checkpoint=True)

maes = []

for i in range(5, 100, 5):
    test_prediction = KNNUserUser(max_neighbors=i).evaluate(r_centered, test,
                                                             load_checkpoint_path="./sim_matrices/sim_matrix_user_20_"
                                                                                  "zscore_no_significance.npy")

    r_test = generate_ratings_matrix(test.values)
    r_test_predicted = z_score_norm.reverse_transform(generate_ratings_matrix(test_prediction.values))

    maes.append(np.nanmean(np.abs(r_test - r_test_predicted)))

    # print("MAE: ", np.nanmean(np.abs(r_test - r_test_predicted)))

plt.plot(maes)
plt.show()
