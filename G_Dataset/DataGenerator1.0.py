import numpy as np
import pandas as pd

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

        noise_types = ['normal', 'uniform', 'poisson']  # 噪声类型列表
        chosen_noise = np.random.choice(noise_types)  # 随机选择一种噪声类型
        feature_range = np.max(X[:, feature_idx]) - np.min(X[:, feature_idx])  # 计算特征的范围

        # 使用用户定义的 noise_range 缩放噪声强度
        noise_strength = feature_range * np.random.uniform(*self.noise_range)

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

    def nonlinear(self, function='sin', add_noise=True):
        # function(str): 非线性函数类型('sin', 'square', 'exp')

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

        data = np.hstack((X, y.reshape(-1, 1)))
        columns = [f"Feature_{i + 1}" for i in range(X.shape[1])] + ["Target"]
        df = pd.DataFrame(data, columns=columns)  # 创建DataFrame
        df.to_csv(filename, index=False)  # 保存为CSV文件
        print(f"Dataset saved to {filename}")

data_gen = DataGenerator(n_samples=1000, n_features=3, noise_range=(0.001, 0.03), feature_ranges=[(0, 10), (-5, 5), (5, 15)], random_seed=42)

# 生成非线性数据集，并添加噪声
X, y = data_gen.nonlinear(function='sin', add_noise=True)

# 保存为CSV文件
data_gen.save2csv(X, y, filename="dataset_nl5x1100_noise.csv")