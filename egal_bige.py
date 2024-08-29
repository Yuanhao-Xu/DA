import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class EGAL:
    def __init__(self, b_factor=0.1):
        self.X_train_labeled_df = None
        self.X_train_unlabeled_df = None
        self.X_train_full_df = None
        self.unlabeled_indices = None
        self.labeled_indices = None
        self.addendum_size = None
        self.b_factor = b_factor
        self.beta = None

    def fit(self, X_labeled, y_labeled):
        pass  # do nothing


    def calculate_similarity_matrix(self):
        # 计算整个训练集的相似度矩阵
        train_similarity_matrix = cosine_similarity(self.X_train_full_df)
        return train_similarity_matrix

    def calculate_alpha_beta(self):
        # 计算相似度矩阵中的非对角元素
        train_similarity_matrix = self.calculate_similarity_matrix()
        similarity_values = train_similarity_matrix[np.triu_indices_from(train_similarity_matrix, k=1)]

        # 计算平均相似度 (mu) 和标准差 (delta)
        mu = np.mean(similarity_values)
        delta = np.std(similarity_values)

        # 计算 alpha
        alpha = mu - 0.5 * delta

        # 将 beta 设置为与 alpha 相同的初始值
        self.beta = alpha
        # print(f"Alpha (α): {alpha}")
        # print(f"Initial Beta (β): {self.beta}")
        return alpha

    def update_beta(self, unlabeled_to_labeled_similarity_matrix):
        # 计算未标记样本与已标记样本的最大相似度
        max_similarities = np.max(unlabeled_to_labeled_similarity_matrix, axis=1)

        # 对这些相似度进行排序
        sorted_similarities = np.sort(max_similarities)

        # 计算新的 beta 值
        index = int(np.floor(self.b_factor * len(sorted_similarities)))
        self.beta = sorted_similarities[index]
        return self.beta

    def select_candidates(self, alpha):
        # 计算未标记数据集与已标记数据集之间的余弦相似度矩阵
        unlabeled_to_labeled_similarity_matrix = cosine_similarity(self.X_train_unlabeled_df, self.X_train_labeled_df)

        # 找到每个未标记样本与其最近的已标记样本之间的最大相似度
        max_similarities = np.max(unlabeled_to_labeled_similarity_matrix, axis=1)

        # 根据初始 beta 的值筛选候选集
        candidate_indices = [idx for idx, similarity in zip(self.unlabeled_indices, max_similarities) if
                             similarity <= self.beta]

        # 确保候选集的大小不少于 addendum_size
        while len(candidate_indices) < self.addendum_size:
            self.b_factor = min(1.0, self.b_factor + 0.1)  # 每次增加 0.1 来扩大候选集
            self.beta = self.update_beta(unlabeled_to_labeled_similarity_matrix)
            candidate_indices = [idx for idx, similarity in zip(self.unlabeled_indices, max_similarities) if
                                 similarity <= self.beta]

        # print("Candidate Set Indices:")
        # print(candidate_indices)
        return candidate_indices

    def calculate_density(self, candidate_indices, alpha):
        candidate_density_scores = []
        for candidate_idx in candidate_indices:
            # 获取候选样本与所有训练集样本的相似度
            similarities = cosine_similarity([self.X_train_unlabeled_df.loc[candidate_idx]],
                                             self.X_train_full_df).flatten()
            # 找到满足 sim(x_i, x_r) >= alpha 的邻域 N_i
            neighborhood = similarities[similarities >= alpha]
            # 计算密度
            density = np.sum(neighborhood)
            candidate_density_scores.append(density)
        return candidate_density_scores

    def calculate_diversity(self, candidate_indices):
        candidate_diversity_scores = []
        for candidate_idx in candidate_indices:
            # 获取候选样本与已标记样本集的相似度
            similarities_to_labeled = cosine_similarity(
                [self.X_train_unlabeled_df.loc[candidate_idx]], self.X_train_labeled_df
            ).flatten()
            # 计算多样性（相似度的倒数）
            diversity = 1 / np.max(similarities_to_labeled)
            candidate_diversity_scores.append(diversity)
        return candidate_diversity_scores

    def combine_density_and_diversity(self, density_scores, diversity_scores, w=0.25):
        # 结合密度和多样性得分
        combined_scores = w * np.array(density_scores) + (1 - w) * np.array(diversity_scores)
        return combined_scores

    def query(self, X_unlabeled, X_labeled, n_act=1, **kwargs):
        self.X_train_labeled_df = X_labeled
        self.X_train_unlabeled_df = X_unlabeled
        self.X_train_full_df = pd.concat([X_labeled, X_unlabeled])
        self.unlabeled_indices = X_unlabeled.index.to_list()
        self.labeled_indices = X_labeled.index.to_list()
        self.addendum_size = n_act
        # Step 1: 计算 alpha 和 beta
        alpha = self.calculate_alpha_beta()

        # Step 2: 选择候选集
        candidate_indices = self.select_candidates(alpha)

        # Step 3: 计算密度和多样性
        density_scores = self.calculate_density(candidate_indices, alpha)
        diversity_scores = self.calculate_diversity(candidate_indices)

        # Step 4: 结合密度和多样性
        combined_scores = self.combine_density_and_diversity(density_scores, diversity_scores, w=0.25)

        # Step 5: 选择得分最高的样本
        top_indices = np.argsort(combined_scores)[-self.addendum_size:]  # 获取分数最高的 addendum_size 个样本索引
        selected_indices = [candidate_indices[i] for i in top_indices]
        # print(f"Selected indices based on combined score: {selected_indices}")
        return selected_indices
