import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor


class GSBAG(GaussianProcessRegressor):
    def __init__(self, random_state=None, kernel=None, **kwargs):
        super().__init__(kernel=kernel, random_state=random_state)
        self.random_state = random_state

    def query(self, X_unlabeled, X_labeled, n_act=1, **kwargs):
        if X_labeled is None:
            raise ValueError("You have to fit it at least once first")
        # 声明一个空的dataframe，用于存储X_unlabeled中分割出来的数据点
        selected_X = pd.DataFrame(columns=X_unlabeled.columns)
        selected_indices = []
        # Here we handle the combination of X correctly
        sigma2_epsilon = self.kernel_.k2.noise_level
        for i in range(n_act):
            combined_X = pd.concat([X_labeled, selected_X])
            K = self.kernel_(combined_X.values.astype(float), combined_X.values.astype(float))
            K_inv = np.linalg.inv(K + np.eye(K.shape[0]) * sigma2_epsilon)
            pi_star_values = []
            for x in X_unlabeled.values.astype(float):
                x = x.reshape(1, -1)
                k_x = self.kernel_(combined_X.values.astype(float), x)
                k_xx = self.kernel_(x, x)
                pi_star = k_xx - k_x.T @ K_inv @ k_x
                pi_star_values.append(pi_star.squeeze())
            pi_star_values = np.array(pi_star_values)
            selected_idx = np.argmax(pi_star_values)

            selected_induce = X_unlabeled.index[selected_idx]
            selected_X = pd.concat([selected_X, X_unlabeled.loc[[selected_induce]]])
            X_unlabeled = X_unlabeled.drop(selected_induce)
            selected_indices.append(selected_induce)

        return selected_indices
