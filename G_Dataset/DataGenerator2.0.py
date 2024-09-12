import numpy as np
import pandas as pd

class DataGenerator:
    def __init__(self, n_samples=1000, n_features=1, noise=0.05, seed=50, add_phase=False):
        """
        初始化 DataGenerator 类，设置样本数、特征数、噪声因子和随机种子。
        add_phase 参数控制是否添加相位偏移，默认不添加。
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.seed = seed
        self.add_phase = add_phase

    def generate_data(self):
        """
        生成非线性回归数据集，可以选择是否添加不同特征的相位偏移。
        """
        # 设置随机种子（如果指定了种子）
        np.random.seed(self.seed)

        # 生成随机的特征矩阵 X，取值范围为 [0, 2π]
        X = np.random.rand(self.n_samples, self.n_features) * 2 * np.pi

        if self.add_phase:
            # 生成随机的相位偏移，每个特征一个
            phase_shifts = np.random.rand(self.n_features) * 2 * np.pi
            # 生成目标变量 y，使用带相位偏移的正弦函数作为非线性关系
            y = np.sum(np.sin(X + phase_shifts), axis=1)
        else:
            # 不使用相位偏移，直接生成目标变量 y
            y = np.sum(np.sin(X), axis=1)

        # 计算目标变量的标准差
        y_std = np.std(y)

        # 动态调整噪声，基于目标变量的标准差
        noise = np.random.normal(0, self.noise * y_std, size=y.shape)

        # 给目标变量加上噪声
        y += noise

        # 将数据转换为 Pandas DataFrame 形式
        self.data = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(self.n_features)])
        self.data['target'] = y

    def save_to_csv(self, file_name):
        """
        将生成的数据保存为CSV文件。
        """
        if hasattr(self, 'data'):
            self.data.to_csv(file_name, index=False)
            print(f"Data saved to {file_name}")
        else:
            print("No data generated. Please run generate_data() first.")


# 设置参数
n_samples = 1100
feature_list = [3, 5, 7, 9, 11]
noise_list_1 = [0.005]  # 第一个任务：特征数为 3、5、7、9、11，噪声为 0.005
noise_list_2 = [0, 0.005, 0.01, 0.015, 0.020]  # 第二个任务：特征数为 5，噪声依次为 0, 0.005, 0.01, 0.015, 0.020

# 任务一：依次生成特征数为 3, 5, 7, 9, 11，噪声为 0.005，默认不添加相位
for n_features in feature_list:
    for noise in noise_list_1:
        generator = DataGenerator(n_samples=n_samples, n_features=n_features, noise=noise, add_phase=False)
        generator.generate_data()
        file_name = f"data_{n_samples}s_{n_features}f_{noise}n.csv"
        generator.save_to_csv(file_name=file_name)

# 任务二：生成特征数为 5，噪声依次为 0, 0.005, 0.01, 0.015, 0.020，选择添加相位
n_features = 5  # 固定特征数为 5
for noise in noise_list_2:
    generator = DataGenerator(n_samples=n_samples, n_features=n_features, noise=noise, add_phase=False)
    generator.generate_data()
    file_name = f"data_{n_samples}s_{n_features}f_{noise}n.csv"
    generator.save_to_csv(file_name=file_name)

