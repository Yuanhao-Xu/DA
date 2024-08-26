import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader, Subset


class DataProcessor:
    def __init__(self, file_path, addendum_init=100, batch_size=32):
        self.file_path = file_path
        self.addendum_init = addendum_init
        self.batch_size = batch_size

        # 加载数据
        self.data = pd.read_excel(self.file_path)

        # 数据标准化
        self.X = self.data.iloc[:, :-1].values  # 转化为 NumPy 数组
        self.y = self.data.iloc[:, -1].values.reshape(-1, 1)  # 目标变量转化为 NumPy 数组并调整为二维形状

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        self.X = scaler_X.fit_transform(self.X)
        self.y = scaler_y.fit_transform(self.y)

        # 转换为 DataFrame 格式并声明行和列索引
        self.X = pd.DataFrame(self.X, columns=self.data.columns[:-1], index=self.data.index)
        self.y = pd.DataFrame(self.y, columns=[self.data.columns[-1]], index=self.data.index)

        # 初始分成训练集和测试集
        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # 获取训练集的索引
        self.train_indices = self.X_train_full.index

        # 定义初始标签集的大小并选择标签索引
        self.labeled_indices = np.random.choice(self.train_indices, size=self.addendum_init, replace=False).tolist()

        # 从训练集索引中去掉标签索引，建立未标记的索引
        self.unlabeled_indices = [idx for idx in self.train_indices if idx not in self.labeled_indices]

        # 根据索引拆分成标签集和非标签集
        self.X_train_labeled_df = self.X_train_full.loc[self.labeled_indices]
        self.y_train_labeled_df = self.y_train_full.loc[self.labeled_indices]

        self.X_train_unlabeled_df = self.X_train_full.loc[self.unlabeled_indices]
        self.y_train_unlabeled_df = self.y_train_full.loc[self.unlabeled_indices]

        # 将完整的训练集和测试集转换为张量
        self.X_train_full_tensor = torch.tensor(self.X_train_full.values, dtype=torch.float32)
        self.y_train_full_tensor = torch.tensor(self.y_train_full.values, dtype=torch.float32)

        self.X_test_tensor = torch.tensor(self.X_test.values, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(self.y_test.values, dtype=torch.float32)

        # 规划测试集
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # 规划训练集
        self.train_full_dataset = TensorDataset(self.X_train_full_tensor, self.y_train_full_tensor)
        self.labeled_subset = Subset(self.train_full_dataset, self.labeled_indices)
        self.train_loader = DataLoader(self.labeled_subset, batch_size=self.batch_size, shuffle=True)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader


# 使用示例
data_processor = DataProcessor(file_path='Dataset/UCI_Concrete_Data.xls', addendum_init=100, batch_size=32)
train_loader = data_processor.get_train_loader()
test_loader = data_processor.get_test_loader()
