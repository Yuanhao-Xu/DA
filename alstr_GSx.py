import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def distance(sample1, sample2):
    # Calculate Euclidean distance between samples
    return np.linalg.norm(sample1 - sample2)


class GSx(XGBRegressor):

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)
        self.random_state = random_state

    def query(self, X_unlabeled, n_act, **kwargs):
        # Compute the sum of distances for each point to all other points

        distances = np.zeros(X_unlabeled.shape[0])
        for i in range(X_unlabeled.shape[0]):
            for j in range(X_unlabeled.shape[0]):
                distances[i] += distance(X_unlabeled.iloc[i].values, X_unlabeled.iloc[j].values)

        # Select the point with the smallest sum of distances as the centroid
        centroid = np.argmin(distances)

        # Get the original index of the first sample in the DataFrame
        first_sample_index = X_unlabeled.index[centroid]
        # Initialize the set of selected samples
        selected_samples_indices = []
        selected_samples_indices.append(first_sample_index)
        # Select the sample closest to the centroid as the first sample
        remaining_samples = X_unlabeled.drop(first_sample_index)

        # Select subsequent samples
        while len(selected_samples_indices) < n_act:
            max_distance = -1
            next_sample_index = None
            for idx, sample in remaining_samples.iterrows():
                # Calculate the minimum distance to any sample in the selected set
                min_distance = min([distance(sample.values, X_unlabeled.loc[i].values) for i in selected_samples_indices])
                # Find the sample with the greatest minimum distance
                if min_distance > max_distance:
                    max_distance = min_distance
                    next_sample_index = idx
            selected_samples_indices.append(next_sample_index)
            remaining_samples = remaining_samples.drop(next_sample_index)

        return selected_samples_indices
