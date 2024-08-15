# CreatTime 2024/8/15

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EGAL:
    def __init__(self, addendum_size=50, w=0.25):
        """
        初始化EGAL算法采样器

        参数:
        addendum_size: int - 需要选择的样本数量，默认值为50
        w: float - 密度和多样性结合时的权重，默认值为0.25
        """
        self.addendum_size = addendum_size
        self.w = w
        self.alpha = None
        self.beta = None

    def calculate_alpha(self, similarity_matrix):
        """
        根据相似度矩阵计算 alpha 值
        用于确定样本的邻域

        参数:
        similarity_matrix: np.ndarray - 样本间的相似度矩阵

        返回:
        alpha: float - 计算出的 alpha 值
        """
        # 提取相似度矩阵的非对角元素
        n = similarity_matrix.shape[0]
        non_diag_elements = similarity_matrix[np.triu_indices(n, k=1)]

        # 计算均值和标准差
        mu = np.mean(non_diag_elements)
        delta = np.std(non_diag_elements)

        # 论文中采用的计算公式 alpha: alpha = mu - 0.5 * delta
        alpha = mu - 0.5 * delta

        return alpha

    def update_beta(self, similarity_matrix, unlabeled_indices, labeled_indices):
        """
        动态更新 beta 值的方法

        参数:
        similarity_matrix: np.ndarray - 样本间的相似度矩阵
        unlabeled_indices: np.ndarray - 未标记样本的索引
        labeled_indices: np.ndarray - 已标记样本的索引

        返回:
        new_beta: float - 更新后的 beta 值
        """
        nearest_similarities = [np.max(similarity_matrix[index, labeled_indices]) for index in unlabeled_indices]
        sorted_similarities = np.sort(nearest_similarities)

        # 初始取前25%分位数
        increment_step = 0.25
        current_step = increment_step
        new_beta = None

        while current_step <= 1.0:
            split_index = int(current_step * len(sorted_similarities))
            S1 = sorted_similarities[:split_index]

            # 如果S1的长度小于addendum_size，那么取更大范围的分位数
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
        """
        计算样本的密度
        样本的密度定义为其邻域内所有样本与该样本的相似度之和

        参数:
        index: int - 样本的索引
        similarity_matrix: np.ndarray - 样本间的相似度矩阵

        返回:
        density: float - 该样本的密度值
        """
        neighbors = similarity_matrix[index] >= self.alpha
        density = np.sum(similarity_matrix[index, neighbors])
        return density

    def calculate_diversity(self, index, similarity_matrix, labeled_indices):
        """
        计算样本的多样性

        参数:
        index: int - 样本的索引
        similarity_matrix: np.ndarray - 样本间的相似度矩阵
        labeled_indices: np.ndarray - 已标记样本的索引

        返回:
        diversity: float - 该样本的多样性值
        """
        nearest_labeled_similarity = np.max(similarity_matrix[index, labeled_indices])
        diversity = 1.0 / nearest_labeled_similarity
        return diversity

    def sample(self, X, labeled_indices, unlabeled_indices):
        """
        执行EGAL算法进行样本选择

        参数:
        X: pd.DataFrame - 特征矩阵
        labeled_indices: np.ndarray - 已标记样本的索引
        unlabeled_indices: np.ndarray - 未标记样本的索引

        返回:
        np.ndarray - 选择的样本在原始 DataFrame 中的索引值
        """
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(X)

        # 根据相似度矩阵计算 alpha 和 beta
        self.alpha = self.calculate_alpha(similarity_matrix)
        self.beta = self.alpha  # 将 beta 初始值设置为 alpha

        candidate_set = []

        # 修改后的部分：确保候选样本集中的样本数至少为 addendum_size
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
        """
        根据样本的相对索引获取在原始DataFrame中的行索引

        参数:
        df: pd.DataFrame - 原始数据集
        top_samples_indices: np.ndarray - 选择的样本的相对索引

        返回:
        np.ndarray - 选择的样本在原始 DataFrame 中的行索引
        """
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

    # print(f"得分最高的 {sampler.addendum_size} 个样本在原始DataFrame中的索引: \n{original_indices.tolist()}")

