import numpy as np
import pandas as pd

class DataGenerator:
    def __init__(self, n_samples=1000, n_features=1, n_lin=0, noise=0.05, seed=50, phase=False):
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

    def generate_data(self):
        """
        生成数据集，其中部分特征是线性关系，剩余特征为非线性（正弦函数）。
        """
        # 设置随机种子
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

        # 计算目标变量的标准差并添加噪声
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

#
# n_samples = 1100
# feature_list = [3, 5, 7, 9, 11]
# noise_list_1 = [0.05]  # 第一个任务：特征数为 3、5、7、9、11，噪声为 0.005
# noise_list_2 = [0, 0.05, 0.1, 0.15, 0.20]  # 第二个任务：特征数为 5，噪声依次为 0, 0.005, 0.01, 0.015, 0.020
#
# # 任务一：依次生成特征数为 3, 5, 7, 9, 11，噪声为 0.005
# for n_features in feature_list:
#     for noise in noise_list_1:
#         # 逻辑：当特征数大于等于7时，将多余的特征定义为线性特征
#         if n_features >= 7:
#             n_lin = n_features - 7  # 超过7的特征定义为线性
#         else:
#             n_lin = 0  # 特征数小于7时，所有特征都是非线性
#
#         # 实例化DataGenerator类
#         generator = DataGenerator(n_samples=n_samples,
#                                   n_features=n_features,
#                                   n_lin=n_lin,
#                                   noise=noise,
#                                   phase=False)
#         generator.generate_data()
#         file_name = f"data_{n_samples}s_{n_features}f{int(noise*100)}n.csv"
#         generator.save_to_csv(file_name)
#
# # 任务二：生成特征数为 5，噪声依次为 0, 0.005, 0.01, 0.015, 0.020
# n_features = 5  # 固定特征数为 5
# for noise in noise_list_2:
#     generator = DataGenerator(n_samples=n_samples,
#                               n_features=n_features,
#                               n_lin=0,  # 特征数小于7，所有特征都是非线性
#                               noise=noise,
#                               phase=False)
#     generator.generate_data()
#     file_name = f"data_{n_samples}s_{n_features}f{int(noise*100)}n.csv"
#     generator.save_to_csv(file_name)




n_features = 5  # 固定特征数为 5

generator = DataGenerator(n_samples=1100,
                          n_features=n_features,
                          n_lin=0,  # 特征数小于7，所有特征都是非线性
                          noise=0.4,
                          phase=False)
generator.generate_data()
file_name = f"40zaosheng.csv"
generator.save_to_csv(file_name)