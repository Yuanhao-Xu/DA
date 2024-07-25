# Author: 
# CreatTime: 2024/7/24
# FileName：LL4AL_main
import random

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, Subset

###
from LL4AL_seed import set_seed
from NN_models import MainNet
from NN_models import LossNet





# 参数
seed = 50
path = '../Dataset/UCI_Concrete_Data.xls'

NUM_TRAIN = 1300 # N 已改
NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 32 # B 已改，主动学习一批的数量
SUBSET    = 50 # M 已改，每次主动学习循环加入的样本数
ADDENDUM  = 50 # K 已改，每次采样的个数
ADDENDUM_init  = 100  # 自定义，初始数据集长度,

MARGIN = 0.7 # xi
WEIGHT = 1.5 # lambda

TRIALS = 3
CYCLES = 14 # 已改，主动学习循环次数

EPOCH = 200
LR = 0.001
MILESTONES = [160]
EPOCHL = 30 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

MOMENTUM = 0.9
WDECAY = 5e-4
# 主程序
def LL4AL_main():
    set_seed(seed)
    device = torch.device('cpu')
    # 加载数据
    file_path = 'Dataset/UCI_Concrete_Data.xls'
    data = pd.read_excel(file_path)

    # 数据标准化
    X = data.iloc[:, :-1].values  # 转化np数组
    y = data.iloc[:, -1].values.reshape(-1, 1)  # 目标变量
    # 实例
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    # 标准化
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # 初始分成训练集和测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换为PyTorch张量
    X_train_full_tensor = torch.tensor(X_train_full, dtype=torch.float32)
    y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    # 规划测试集
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
    # TODO 规划训练集（暂时）
    train_full_dataset = TensorDataset(X_train_full_tensor, y_train_full_tensor)
    train_loader = DataLoader(train_full_dataset, batch_size=BATCH, shuffle=True)

    plt.ion()

    indices = list(range(len(X_train_full)))
    random.shuffle(indices)
    # 标签数据和为标签数据的索引号
    labeled_set = indices[:ADDENDUM_init]  # 初始数据集长度
    unlabeled_set = indices[ADDENDUM_init:]

    # 把整个训练集划分为标签子集和非标签子集
    labeled_subset = Subset(train_full_dataset, labeled_set)
    unlabeled_subset = Subset(train_full_dataset, unlabeled_set)

    # # 分离特征和标签
    # X_initial, y_initial = labeled_subset
    # X_pool, y_pool = unlabeled_subset

    # 建立标签子集的训练集
    # train_dataset = TensorDataset(X_initial, y_initial)
    train_loader = DataLoader(labeled_subset, batch_size=BATCH, shuffle=True)




