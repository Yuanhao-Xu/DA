import random

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, Subset

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(50) # 42




class DataProcessor:
    def __init__(self, file_path, addendum_init, batch_size=32):
        self.file_path = file_path
        self.addendum_init = addendum_init
        self.batch_size = batch_size


        # 加载数据
        self.data = pd.read_csv(self.file_path)

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

        # 在划分训练集和测试集后，重置它们的索引，因为实际上后续抽取标记/非标记数据集仅是在训练集上操作的
        self.X_train_full = self.X_train_full.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_train_full = self.y_train_full.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)

        # DataFrame形式的测试集特征/标签
        self.X_test_df = self.X_test
        self.y_test_df = self.y_test

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




__all__ = [
    'train_loader',
    'test_loader',
    'labeled_indices',
    'unlabeled_indices',
    'train_full_dataset',
    'X_train_labeled_df',
    'y_train_labeled_df',
    'X_train_unlabeled_df',
    'y_train_unlabeled_df',
    'X_train_full_df',
    'y_train_full_df',
    'X_test_df',
    'y_test_df'
]



paths = {"UCI":"Dataset/concrete/concrete_data.csv",
         "BFRC_cs":"Dataset/BFRC/data_cs.csv",
         "BFRC_fs":"Dataset/BFRC/data_fs.csv",
         "BFRC_sts":"Dataset/BFRC/data_sts.csv",
         "pullout_fmax":"Dataset/pullout/dataset_fmax.csv",
         "pullout_ifss":"Dataset/pullout/dataset_ifss.csv",
         "ENB2012":"Dataset/ENB2012/data1.csv",
         "GEN3f5n":"G_Dataset/data_1100s_3f5n.csv",
         "GEN5f5n":"G_Dataset/data_1100s_3f5n.csv",# 跑错

         "GEN7f5n":"G_Dataset/data_1100s_7f5n.csv",
         "GEN5f0n":"G_Dataset/data_1100s_5f_0n.csv",
         "GEN5f20n":"G_Dataset/data_1100s_5f20n.csv",
         "测试":"G_Dataset/40zaosheng.csv"


         }


# 实例化 DataProcessor 类
Dataset_UCI = DataProcessor(file_path=paths["GEN7f5n"], addendum_init=10)

# 获取训练器
train_loader = Dataset_UCI.train_loader
test_loader = Dataset_UCI.test_loader

# 获取标签和非标签索引
labeled_indices = Dataset_UCI.labeled_indices
unlabeled_indices = Dataset_UCI.unlabeled_indices

# 获取完整训练集
train_full_dataset = Dataset_UCI.train_full_dataset

# 获取标记/未标记数据集的特征/标签
X_train_labeled_df = Dataset_UCI.X_train_labeled_df
y_train_labeled_df = Dataset_UCI.y_train_labeled_df

X_train_unlabeled_df = Dataset_UCI.X_train_unlabeled_df
y_train_unlabeled_df = Dataset_UCI.y_train_unlabeled_df

# 获取完整训练集的特征和标签
X_train_full_df = Dataset_UCI.X_train_full
y_train_full_df = Dataset_UCI.y_train_full

# 用于xgb
X_test_df = Dataset_UCI.X_test_df
y_test_df = Dataset_UCI.y_test_df


