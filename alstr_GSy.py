import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def distance(sample1, sample2):
    # Calculate Euclidean distance between samples
    return np.linalg.norm(sample1 - sample2)


class GSy(XGBRegressor):

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)
        self.random_state = random_state

    def query(self, X_unlabeled, n_act, X_labeled, y_labeled, y_unlabeled, **kwargs):
        XGB = XGBRegressor(n_estimators=100, random_state=self.random_state)
        selected_indices = []

        while len(selected_indices) < n_act:
            XGB.fit(X_labeled, y_labeled)
            y_predict = XGB.predict(X_unlabeled)
            y_predict = pd.DataFrame(y_predict, index=X_unlabeled.index)
            max_distance = -1
            next_sample_index = None

            for idx, sample in y_predict.iterrows():
                # Calculate the minimum distance to any sample in the selected set
                min_distance = min([distance(sample.values, y_labeled.loc[i].values) for i in y_labeled.index])
                # Find the sample with the greatest minimum distance
                if min_distance > max_distance:
                    max_distance = min_distance
                    next_sample_index = idx
            selected_indices.append(next_sample_index)
            X_labeled = pd.concat([X_labeled, X_unlabeled.loc[[next_sample_index]]])
            X_unlabeled = X_unlabeled.drop(next_sample_index)
            y_labeled = pd.concat([y_labeled, y_unlabeled.loc[[next_sample_index]]])
            y_unlabeled = y_unlabeled.drop(next_sample_index)

        return selected_indices
