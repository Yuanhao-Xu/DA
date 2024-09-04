from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EGAL:
    def __init__(self):
        """
        初始化 EGAL 对象，不需要传递数据参数。
        """
        self.beta = None

    def calculate_similarity_matrix(self, X_train_full_df):
        # 计算整个训练集的相似度矩阵
        train_similarity_matrix = cosine_similarity(X_train_full_df)
        return train_similarity_matrix

    def calculate_alpha_beta(self, X_train_full_df):
        # 计算相似度矩阵中的非对角元素
        train_similarity_matrix = self.calculate_similarity_matrix(X_train_full_df)
        similarity_values = train_similarity_matrix[np.triu_indices_from(train_similarity_matrix, k=1)]

        # 计算平均相似度 (mu) 和标准差 (delta)
        mu = np.mean(similarity_values)
        delta = np.std(similarity_values)

        # 计算 alpha
        alpha = mu - 0.5 * delta

        # 将 beta 设置为与 alpha 相同的初始值
        self.beta = alpha
        return alpha

    def update_beta(self, unlabeled_to_labeled_similarity_matrix, b_factor):
        # 计算未标记样本与已标记样本的最大相似度
        max_similarities = np.max(unlabeled_to_labeled_similarity_matrix, axis=1)

        # 对这些相似度进行排序
        sorted_similarities = np.sort(max_similarities)

        # 计算新的 beta 值
        index = min(int(np.floor(b_factor * len(sorted_similarities))), len(sorted_similarities)-1)
        self.beta = sorted_similarities[index]
        return self.beta

    def select_candidates(self, X_train_unlabeled_df, X_train_labeled_df, unlabeled_indices, addendum_size, alpha, b_factor):
        # 计算未标记数据集与已标记数据集之间的余弦相似度矩阵
        unlabeled_to_labeled_similarity_matrix = cosine_similarity(X_train_unlabeled_df, X_train_labeled_df)

        # 找到每个未标记样本与其最近的已标记样本之间的最大相似度
        max_similarities = np.max(unlabeled_to_labeled_similarity_matrix, axis=1)

        # 根据初始 beta 的值筛选候选集
        candidate_indices = [idx for idx, similarity in zip(unlabeled_indices, max_similarities) if
                             similarity <= self.beta]

        # 确保候选集的大小不少于 addendum_size
        while len(candidate_indices) < addendum_size:
            b_factor = min(1.0, b_factor + 0.1)  # 每次增加 0.1 来扩大候选集
            self.beta = self.update_beta(unlabeled_to_labeled_similarity_matrix, b_factor)
            candidate_indices = [idx for idx, similarity in zip(unlabeled_indices, max_similarities) if
                                 similarity <= self.beta]

        return candidate_indices

    def calculate_density(self, X_train_unlabeled_df, X_train_full_df, candidate_indices, alpha):
        candidate_density_scores = []
        for candidate_idx in candidate_indices:
            # 获取候选样本与所有训练集样本的相似度
            similarities = cosine_similarity([X_train_unlabeled_df.loc[candidate_idx]],
                                             X_train_full_df).flatten()
            # 找到满足 sim(x_i, x_r) >= alpha 的邻域 N_i
            neighborhood = similarities[similarities >= alpha]
            # 计算密度
            density = np.sum(neighborhood)
            candidate_density_scores.append(density)
        return candidate_density_scores

    def calculate_diversity(self, X_train_unlabeled_df, X_train_labeled_df, candidate_indices):
        candidate_diversity_scores = []
        for candidate_idx in candidate_indices:
            # 获取候选样本与已标记样本集的相似度
            similarities_to_labeled = cosine_similarity(
                [X_train_unlabeled_df.loc[candidate_idx]], X_train_labeled_df
            ).flatten()
            # 计算多样性（相似度的倒数）
            diversity = 1 / np.max(similarities_to_labeled)
            candidate_diversity_scores.append(diversity)
        return candidate_diversity_scores

    def combine_density_and_diversity(self, density_scores, diversity_scores, w=0.25):
        # 结合密度和多样性得分
        combined_scores = w * np.array(density_scores) + (1 - w) * np.array(diversity_scores)
        return combined_scores

    def query(self, X_train_labeled_df, X_train_unlabeled_df, X_train_full_df, addendum_size, b_factor=0.1):
        """
        查询方法，用于选择下一个未标记样本集。

        参数:
        - X_train_labeled_df: 已标记训练数据的特征 DataFrame。
        - X_train_unlabeled_df: 未标记训练数据的特征 DataFrame。
        - X_train_full_df: 全部训练数据的特征 DataFrame。
        - addendum_size: 本次选择的样本数量。
        - b_factor: 控制候选集大小的初始比例。

        返回:
        - selected_indices (list): 选择的未标记数据的索引列表。
        """
        self.unlabeled_indices = X_train_unlabeled_df.index.tolist()
        self.labeled_indices = X_train_labeled_df.index.tolist()

        # Step 1: 计算 alpha 和 beta
        alpha = self.calculate_alpha_beta(X_train_full_df)

        # Step 2: 选择候选集
        candidate_indices = self.select_candidates(X_train_unlabeled_df, X_train_labeled_df, self.unlabeled_indices, addendum_size, alpha, b_factor)

        # Step 3: 计算密度和多样性
        density_scores = self.calculate_density(X_train_unlabeled_df, X_train_full_df, candidate_indices, alpha)
        diversity_scores = self.calculate_diversity(X_train_unlabeled_df, X_train_labeled_df, candidate_indices)

        # Step 4: 结合密度和多样性
        combined_scores = self.combine_density_and_diversity(density_scores, diversity_scores, w=0.25)

        # Step 5: 选择得分最高的样本
        top_indices = np.argsort(combined_scores)[-addendum_size:]  # 获取分数最高的 addendum_size 个样本索引
        selected_indices = [candidate_indices[i] for i in top_indices]
        return selected_indices
