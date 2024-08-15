# CreatTime 2024/8/15

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EGAL:
    def __init__(self, addendum_size=50, w=0.25):

        self.addendum_size = addendum_size
        self.w = w
        self.alpha = None
        self.beta = None

    def calculate_alpha(self, similarity_matrix):

        # 提取相似度矩阵的非对角元素
        n = similarity_matrix.shape[0]
        non_diag_elements = similarity_matrix[np.triu_indices(n, k=1)]

        mu = np.mean(non_diag_elements)
        delta = np.std(non_diag_elements)

        # 论文中采用的计算公式 alpha: alpha = mu - 0.5 * delta
        alpha = mu - 0.5 * delta

        return alpha

    def update_beta(self, similarity_matrix, unlabeled_indices, labeled_indices):

        nearest_similarities = [np.max(similarity_matrix[index, labeled_indices]) for index in unlabeled_indices]
        sorted_similarities = np.sort(nearest_similarities)

        # 初始取前25%分位数
        increment_step = 0.25
        current_step = increment_step
        new_beta = None

        while current_step <= 1.0:
            split_index = int(current_step * len(sorted_similarities))
            S1 = sorted_similarities[:split_index]

            # 如果S1的长度小于addendum_size，取更大范围的分位数
            if len(S1) < self.addendum_size:
                current_step += increment_step
            else:
                new_beta = np.max(S1)
                break

        # 如果无法在循环内找到合适的分位数，直接取sorted_similarities的最大值
        if new_beta is None:
            new_beta = np.max(sorted_similarities)

        return new_beta

    def calculate_density(self, index, similarity_matrix):

        neighbors = similarity_matrix[index] >= self.alpha
        density = np.sum(similarity_matrix[index, neighbors])
        return density

    def calculate_diversity(self, index, similarity_matrix, labeled_indices):

        nearest_labeled_similarity = np.max(similarity_matrix[index, labeled_indices])
        diversity = 1.0 / nearest_labeled_similarity
        return diversity

    def sample(self, X, labeled_indices, unlabeled_indices):

        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(X)

        # 根据相似度矩阵计算 alpha 和 beta
        self.alpha = self.calculate_alpha(similarity_matrix)
        self.beta = self.alpha  # 将 beta 初始值设置为 alpha

        candidate_set = []

        # 确保候选样本集中的样本数至少为 addendum_size
        while len(candidate_set) < self.addendum_size:
            candidate_set.clear()  # 清空候选集
            for index in unlabeled_indices:
                nearest_labeled_similarity = np.max(similarity_matrix[index, labeled_indices])
                if nearest_labeled_similarity <= self.beta:
                    candidate_set.append(index)

            # 如果候选样本集的数量少于 addendum_size，更新 beta
            if len(candidate_set) < self.addendum_size:
                self.beta = self.update_beta(similarity_matrix, unlabeled_indices, labeled_indices)

        candidate_set = np.array(candidate_set)

        densities = [self.calculate_density(index, similarity_matrix) for index in candidate_set]
        diversities = [self.calculate_diversity(index, similarity_matrix, labeled_indices) for index in candidate_set]

        combined_scores = self.w * np.array(densities) + (1 - self.w) * np.array(diversities)

        ranked_candidates = [x for _, x in sorted(zip(combined_scores, candidate_set), reverse=True)]

        top_samples_indices = ranked_candidates[:self.addendum_size]

        return top_samples_indices

    def get_original_indices(self, df, top_samples_indices):

        return df.index[top_samples_indices]



if __name__ == "__main__":
    file_path = 'Dataset/UCI_Concrete_Data.xls'
    df = pd.read_excel(file_path)

    X = df.drop(columns=['Concrete compressive strength(MPa, megapascals) '])
    y = df['Concrete compressive strength(MPa, megapascals) ']

    np.random.seed(42)
    labeled_indices = np.random.choice(len(X), size=100, replace=False)
    unlabeled_indices = np.array([i for i in range(len(X)) if i not in labeled_indices])

    sampler = EGAL(addendum_size=50, w=0.25)
    top_samples_indices = sampler.sample(X, labeled_indices, unlabeled_indices)
    original_indices = sampler.get_original_indices(df, top_samples_indices)



