import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def distance(sample1, sample2):
    # 计算样本之间的欧氏距离
    return np.linalg.norm(sample1 - sample2)


class GSx(XGBRegressor):

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)
        self.random_state = random_state

    def query(self, X_unlabeled, n_act, **kwargs):
        # 计算df中每个点与其他所有点的距离的和

        distances = np.zeros(X_unlabeled.shape[0])
        for i in range(X_unlabeled.shape[0]):
            for j in range(X_unlabeled.shape[0]):
                distances[i] += distance(X_unlabeled.iloc[i].values, X_unlabeled.iloc[j].values)

        # 选择距离和最小的点作为第一个点
        centroid = np.argmin(distances)

        # 获得第一个点在df中的标准索引
        first_sample_index = X_unlabeled.index[centroid]
        # 初始化已选择样本集合
        selected_samples_indices = []
        selected_samples_indices.append(first_sample_index)
        # 选择与质心最近的样本作为第一个样本
        remaining_samples = X_unlabeled.drop(first_sample_index)

        # 选择后续样本
        while len(selected_samples_indices) < n_act:
            max_distance = -1
            next_sample_index = None
            for idx, sample in remaining_samples.iterrows():
                # 计算该样本到已选择样本集中每个样本的距离的最小值
                min_distance = min([distance(sample.values, X_unlabeled.loc[i].values) for i in selected_samples_indices])
                # 找出距离最远的那个样本
                if min_distance > max_distance:
                    max_distance = min_distance
                    next_sample_index = idx
            selected_samples_indices.append(next_sample_index)
            remaining_samples = remaining_samples.drop(next_sample_index)

        return selected_samples_indices
