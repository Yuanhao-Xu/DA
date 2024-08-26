import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader, Subset

# 加载数据
file_path = 'Dataset/UCI_Concrete_Data.xls'
data = pd.read_excel(file_path)
# 数据标准化
X = data.iloc[:, :-1].values  # 转化为 NumPy 数组
y = data.iloc[:, -1].values.reshape(-1, 1)  # 目标变量转化为 NumPy 数组并调整为二维形状

# 实例化 MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# 对特征变量和目标变量进行标准化
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 将标准化后的 X 和 y 转换回 DataFrame 格式，并直接声明行和列索引
X = pd.DataFrame(X, columns=data.columns[:-1], index=data.index)
y = pd.DataFrame(y, columns=[data.columns[-1]], index=data.index)

# 初始分成训练集和测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 获取训练集的索引
train_indices = X_train_full.index

# 定义初始标签集的大小
addendum_init = 100  # 可以根据需要调整
labeled_indices = np.random.choice(train_indices, size=addendum_init, replace=False).tolist()
# 从训练集索引中去掉标签索引，建立未标记的索引
unlabeled_indices = [idx for idx in train_indices if idx not in labeled_indices]

# 根据索引拆分成标签集和非标签集
X_train_labeled_df = X_train_full.loc[labeled_indices]
y_train_labeled_df = y_train_full.loc[labeled_indices]

X_train_unlabeled_df = X_train_full.loc[unlabeled_indices]
y_train_unlabeled_df = y_train_full.loc[unlabeled_indices]

# 将完整的训练集和测试集转换为张量
X_train_full_tensor = torch.tensor(X_train_full.values, dtype=torch.float32)
y_train_full_tensor = torch.tensor(y_train_full.values, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# 规划测试集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 规划训练集
batch_size = 32
train_full_dataset = TensorDataset(X_train_full_tensor, y_train_full_tensor)
labeled_subset = Subset(train_full_dataset, labeled_indices)
train_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True)
