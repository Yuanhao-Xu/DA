import numpy as np
import pandas as pd
from sklearn.datasets import make_regression



class DataGenerator:
    def __init__(self, n_samples=100, n_features=1, noise_range=(0.1, 1.0), feature_ranges=None, random_seed=None):
        """
        初始化DataGenerator类。

        参数:
        n_samples (int): 样本数量。
        n_features (int): 特征数量。
        noise_range (tuple): 噪声的范围，针对每个特征随机生成噪声强度。
        feature_ranges (list or None): 每个特征的值范围，格式为[(min1, max1), (min2, max2), ...]。
        random_seed (int or None): 随机种子，用于控制随机性的可重复性。如果为None，则不设置种子。
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_range = noise_range

        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)

        # 如果没有传入特征范围，则默认为[0, 10]范围
        if feature_ranges is None:
            self.feature_ranges = [(0, 10)] * self.n_features
        else:
            if len(feature_ranges) != n_features:
                raise ValueError("feature_ranges length must match n_features")
            self.feature_ranges = feature_ranges

    def _apply_noise(self, X, feature_idx):
        """
        随机为某个特征应用不同的噪声类型，并根据特征的值范围调整噪声强度。

        参数:
        X (ndarray): 特征矩阵。
        feature_idx (int): 特征的索引号。
        """
        noise_types = ['normal', 'uniform', 'poisson']  # 噪声类型列表
        chosen_noise = np.random.choice(noise_types)  # 随机选择一种噪声类型
        feature_range = np.max(X[:, feature_idx]) - np.min(X[:, feature_idx])  # 计算特征的范围

        # 动态调整噪声强度，基于特征范围计算初始噪声强度
        base_noise_strength = feature_range

        # 使用用户定义的 noise_range 缩放噪声强度
        noise_strength = base_noise_strength * np.random.uniform(*self.noise_range)

        if chosen_noise == 'normal':
            # 正态分布噪声，基于特征范围调整噪声强度
            noise = np.random.normal(0, noise_strength, self.n_samples)
        elif chosen_noise == 'uniform':
            # 均匀分布噪声，基于特征范围调整噪声
            noise = np.random.uniform(-noise_strength, noise_strength, self.n_samples)
        elif chosen_noise == 'poisson':
            # 泊松噪声（由于泊松分布是离散的，因此添加一些缩放）
            noise = np.random.poisson(lam=noise_strength, size=self.n_samples) - noise_strength
        else:
            noise = np.zeros(self.n_samples)  # 默认不添加噪声

        # 将噪声应用到特定的特征
        X[:, feature_idx] += noise

    def linear(self, add_noise=True):
        """
        生成线性回归数据集。

        参数:
        add_noise (bool): 是否添加噪声。

        返回:
        X (ndarray): 特征矩阵。
        y (ndarray): 标签数组。
        """
        X = np.zeros((self.n_samples, self.n_features))

        # 为每个特征生成数据
        for i in range(self.n_features):
            X[:, i] = np.random.uniform(self.feature_ranges[i][0], self.feature_ranges[i][1], self.n_samples)

        y = np.dot(X, np.random.rand(self.n_features))  # 简单的线性回归目标

        if add_noise:
            for i in range(self.n_features):
                self._apply_noise(X, i)  # 为每个特征应用不同的噪声类型

        X = np.round(X, 3)  # 保留三位小数
        y = np.round(y, 3)  # 保留三位小数
        return X, y

    def nonlinear(self, function='sin', add_noise=True):
        """
        生成非线性回归数据集。

        参数:
        function (str): 非线性函数类型 ('sin', 'square', 'exp')。
        add_noise (bool): 是否添加噪声。

        返回:
        X (ndarray): 特征矩阵。
        y (ndarray): 标签数组。
        """
        X = np.zeros((self.n_samples, self.n_features))

        # 为每个特征生成数据
        for i in range(self.n_features):
            X[:, i] = np.random.uniform(self.feature_ranges[i][0], self.feature_ranges[i][1], self.n_samples)

        y = np.zeros(self.n_samples)  # 初始化 y

        # 对每个特征应用非线性变换并加总到 y 中
        for i in range(self.n_features):
            if function == 'sin':
                y += np.sin(X[:, i])
            elif function == 'square':
                y += np.power(X[:, i], 2)
            elif function == 'exp':
                y += np.exp(X[:, i])
            else:
                raise ValueError("Unknown function type. Choose 'sin', 'square', or 'exp'.")

        if add_noise:
            for i in range(self.n_features):
                self._apply_noise(X, i)  # 为每个特征应用不同的噪声类型

        # 保留三位小数
        X = np.round(X, 3)
        y = np.round(y, 3)

        return X, y

    def save2csv(self, X, y, filename="dataset.csv"):
        """
        将生成的数据集保存为CSV文件。

        参数:
        X (ndarray): 特征矩阵。
        y (ndarray): 标签数组。
        filename (str): 保存文件名。
        """
        data = np.hstack((X, y.reshape(-1, 1)))  # 将X和y合并为一个矩阵
        columns = [f"Feature_{i + 1}" for i in range(X.shape[1])] + ["Target"]  # 定义列名
        df = pd.DataFrame(data, columns=columns)  # 创建DataFrame
        df.to_csv(filename, index=False)  # 保存为CSV文件
        print(f"Dataset saved to {filename}")


# 为每个特征设置不同的范围
feature_ranges = [(0, 10),(-5, 5),(10, 25)]

# # 实例化类并生成线性回归数据集
# Dataset_l1 = DataGenerator(n_samples=1100, n_features=3, noise_range=(0.01, 0.15), feature_ranges=feature_ranges, random_seed=42)
# X_l1, y_l1 = Dataset_l1.linear(add_noise=False)
# Dataset_l1.save2csv(X_l1, y_l1, filename="dataset_l5x1100.csv")
#
# # 实例化类并生成非线性回归数据集
# Dataset_nl1 = DataGenerator(n_samples=1100, n_features=3, noise_range=(0.01, 0.15), feature_ranges=feature_ranges)
# X_nl1, y_nl1 = Dataset_nl1.nonlinear(function='sin', add_noise=False)
# Dataset_nl1.save2csv(X_nl1, y_nl1, filename="dataset_nl5x1100.csv")
# 无噪声
Dataset_nl1 = DataGenerator(n_samples=1100, n_features=3, noise_range=(0.00, 0.03), feature_ranges=feature_ranges)
X_nl1, y_nl1 = Dataset_nl1.nonlinear(function='sin', add_noise=True)
Dataset_nl1.save2csv(X_nl1, y_nl1, filename="dataset_nl5x1100_noise.csv")
