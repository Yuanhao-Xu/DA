import numpy as np
import pandas as pd

class DataGenerator:
    def __init__(self, n_samples=1000, n_features=1, n_lin=0, noise=0.05, seed=50, phase=True):
        """
        初始化 DataGenerator 类，设置样本数、特征数、线性特征数、噪声因子和随机种子。
        phase 参数控制是否添加相位偏移，默认不添加。
        n_lin 参数控制线性特征的数量。
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_lin = n_lin
        self.noise = noise
        self.seed = seed
        self.phase = phase

        # 确保线性特征数量不超过总特征数
        assert 0 <= self.n_lin <= self.n_features, "线性特征数量不能超过总特征数量"

    def generate_data(self, noise_level=0):
        """
        生成数据集，其中部分特征是线性关系，剩余特征为非线性（正弦函数）。
        noise_level 参数用于生成不同等级的噪声，生成不同噪声等级时使用不同的随机种子。
        """
        # 设置主随机种子，生成特征数据
        np.random.seed(self.seed)

        # 生成随机特征矩阵 X，取值范围为 [0, 2π]，其中部分为线性特征
        X = np.random.rand(self.n_samples, self.n_features) * 2 * np.pi

        # 生成线性特征（前 n_lin 个为线性特征）
        X_linear = X[:, :self.n_lin]

        # 非线性特征使用正弦函数
        X_nonlinear = X[:, self.n_lin:]

        if self.phase:
            # 为非线性特征添加相位偏移
            phase_shifts = np.random.rand(self.n_features - self.n_lin) * 2 * np.pi
            y_nonlinear = np.sum(np.sin(X_nonlinear + phase_shifts), axis=1)
        else:
            # 不使用相位偏移
            y_nonlinear = np.sum(np.sin(X_nonlinear), axis=1)

        # 对线性特征进行线性组合
        y_linear = np.sum(X_linear, axis=1)

        # 总目标变量 y 是线性和非线性部分的组合
        y = y_linear + y_nonlinear

        # 为了生成不同的噪声，使用基于初始随机种子的 noise_level 作为偏移生成新的子种子
        noise_seed = self.seed + noise_level  # 使用不同的种子生成噪声
        np.random.seed(noise_seed)
        y_std = np.std(y)
        noise = np.random.normal(0, self.noise * y_std, size=y.shape)
        y += noise

        # 将数据转换为 Pandas DataFrame
        self.data = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(self.n_features)])
        self.data['target'] = y

        # 保留四位小数
        self.data = self.data.round(4)

    def save_to_csv(self, file_name):
        """
        将生成的数据保存为CSV文件。
        """
        if hasattr(self, 'data'):
            self.data.to_csv(file_name, index=False)
            print(f"Data saved to {file_name}")
        else:
            print("No data generated. Please run generate_data() first.")


# # 生成7个特征 10%噪声的数据集
# gen = DataGenerator(n_samples=1100, n_features=7, n_lin=0, noise=0.10, seed=50)
# gen.generate_data(noise_level=0)
# gen.save_to_csv('data_1100s_7f10n_NEW.csv')
#
# # 生成7个特征 10%噪声的数据集
# gen = DataGenerator(n_samples=1100, n_features=7, n_lin=0, noise=0.15, seed=50)
# gen.generate_data(noise_level=1)
# gen.save_to_csv('data_1100s_7f15n_NEW.csv')
#
# # 生成7个特征 10%噪声的数据集
# gen = DataGenerator(n_samples=1100, n_features=7, n_lin=0, noise=0.20, seed=50)
# gen.generate_data(noise_level=2)
# gen.save_to_csv('data_1100s_7f20n_NEW.csv')

# 生成9个特征 5%噪声的数据集
gen = DataGenerator(n_samples=1100, n_features=9, n_lin=0, noise=0.05, seed=50)
gen.generate_data(noise_level=0)
gen.save_to_csv('data_1100s_9f5n_NEW.csv')

# # 生成11个特征 5%噪声的数据集
# gen = DataGenerator(n_samples=1100, n_features=11, n_lin=0, noise=0.05, seed=50)
# gen.generate_data(noise_level=0)
# gen.save_to_csv('data_1100s_11f5n_NEW.csv')
