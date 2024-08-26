# CreatTime 2024/7/25
# 导入库
import random
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

# 导入自定义变量和函数
from data_process import (train_loader,
                          test_loader,
                          labeled_indices,
                          unlabeled_indices,
                          train_full_dataset,
                          X_train_labeled_df,
                          y_train_labeled_df,
                          X_train_unlabeled_df,
                          y_train_unlabeled_df,
                          X_train_full_df,
                          y_train_full_df)
from benchmark_nn_model import BenchmarkModel, ModelTrainer
from RS.RS_strat import RS
from LL4AL.LL_main_pro import LL4AL
from bmdal_reg.BMDAL_strat import BMDAL
from MC_Dropout.MCD_strat import MCD
from EGAL import EGAL
from BayesianAL import BayesianAL
import pyro

# 设置 Pyro 的随机种子
pyro.set_rng_seed(42)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)


SEED = 50
set_seed(SEED)

# 基准模型参数
strategy = "LL4AL"

addendum_init = 100
BATCH = 32

num_cycles = 14
epochs = 500
addendum_size = 50

NN_input = 8
NN_output = 1

# 基准模型初始化

device = torch.device('cpu')
model = BenchmarkModel(input_dim=NN_input, output_dim=NN_output)
trainer = ModelTrainer(model, device=device, lr=0.001)


### ====================================================主动学习过程
# indices = list(range(len(X_train_full)))
# random.shuffle(indices)
# # 标签数据和无标签数据的索引号
# labeled_set = indices[:addendum_init]  # 初始数据集长度
# unlabeled_set = indices[addendum_init:]


# # 把整个训练集划分为标签子集和非标签子集
# labeled_subset = Subset(train_full_dataset, labeled_set)
# unlabeled_subset = Subset(train_full_dataset, unlabeled_set)

# 定义分离特征和标签的函数
# def split_features_labels(dataset, indices):
#     subset = Subset(dataset, indices)
#     features_list = []
#     labels_list = []
#
#     for i in range(len(subset)):
#         features, label = subset[i]
#         features_list.append(features)
#         labels_list.append(label)
#
#     features_tensor = torch.stack(features_list)
#     labels_tensor = torch.tensor(labels_list)
#
#     return subset, features_tensor, labels_tensor
#
#
# labeled_subset, X_initial, y_initial = split_features_labels(train_full_dataset, labeled_set)
# unlabeled_subset, X_unlabeled, y_unlabeled = split_features_labels(train_full_dataset, unlabeled_set)




# 建立标签子集的训练集
# train_dataset = TensorDataset(X_initial, y_initial)

# ###################################################8.15修改的###################################################
# # 将训练集和测试集转化为DataFrame，以便获取潜在索引
# X_train_full_df = pd.DataFrame(X_train_full, columns=data.columns[:-1])
# X_test_df = pd.DataFrame(X_test, columns=data.columns[:-1])
#
# # 获取训练集和测试集的潜在索引_
# train_indices = X_train_full_df.index.tolist()
# test_indices = X_test_df.index.tolist()
#
# # 随机选择 addendum_init 个索引作为初始标签集
#
# labeled_indices = np.random.choice(train_indices, size=addendum_init, replace=False).tolist()
#
# # 剩余的训练集作为未标记数据集
# unlabeled_indices = list(set(train_indices) - set(labeled_indices))
#
# """
# 将训练数据转换为PyTorch的Tensor
# """
# X_train_labeled = torch.tensor(X_train_full[labeled_indices], dtype=torch.float32)
# y_train_labeled = torch.tensor(y_train_full[labeled_indices], dtype=torch.float32)
# X_train_unlabeled = torch.tensor(X_train_full[unlabeled_indices], dtype=torch.float32)
# y_train_unlabeled = torch.tensor(y_train_full[unlabeled_indices], dtype=torch.float32)
# #
# labeled_subset = Subset(train_full_dataset, labeled_indices)
# train_loader = DataLoader(labeled_subset, batch_size=BATCH, shuffle=True)

test_losses = []
test_R2s = []


###################################################8.15修改的###################################################
for cycle in range(num_cycles):

    print(f"Active Learning Cycle {cycle + 1}/{num_cycles}")

    # 训练模型
    model = trainer.train_model(train_loader, epochs)
    # 测试模型
    test_loss, test_R2 = trainer.evaluate_model(test_loader)

    test_losses.append(test_loss)
    test_R2s.append(test_R2)
    """
    调用一种主动学习策略添加新的数据集索引，并更新标签集和非标签集
    方法：
    1.RS
    2.LL4AL
    3.BMDAL
    4.MCD
    5.EGAL
    6.BayesianAL
    """

    if strategy == "RS":
        selected_indices = RS(unlabeled_indices, addendum_size)
        labeled_indices.extend(selected_indices)
        for idx in selected_indices:
            unlabeled_indices.remove(idx)
        # 创建包含更新后的已标注样本子集
        labeled_subset = Subset(train_full_dataset, labeled_indices)
        # 创建 DataLoader
        train_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
        # [0.709, 0.6615, 0.8432, 0.8359, 0.867, 0.8429, 0.7907, 0.7931, 0.7853, 0.8309, 0.8688, 0.8839, 0.9315, 0.9546]

    if strategy == "LL4AL":# TODO
        learning_loss = LL4AL(BATCH=32, LR=0.001, MARGIN=0.7, WEIGHT=1.5, EPOCH=200, EPOCHL=30, WDECAY=5e-4)
        selected_indices = learning_loss.query(X_train_unlabeled_df, X_train_labeled_df, y_train_unlabeled_df,
                                               y_train_labeled_df, n_act=addendum_size)

        labeled_indices.extend(selected_indices)
        for idx in selected_indices:
            unlabeled_indices.remove(idx)
        # 创建包含更新后的已标注样本子集
        labeled_subset = Subset(train_full_dataset, labeled_indices)
        # 创建 DataLoader
        train_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
        # 索引更新后，传入参数也要更新
        X_train_labeled_df = X_train_full_df.loc[labeled_indices]
        y_train_labeled_df = y_train_full_df.loc[labeled_indices]
        X_train_unlabeled_df = X_train_full_df.loc[unlabeled_indices]
        y_train_unlabeled_df = y_train_full_df.loc[unlabeled_indices]



    if strategy == "BMDAL":
        train_loader, X_initial, y_initial, X_unlabeled, y_unlabeled = BMDAL(X_initial, y_initial, X_unlabeled, y_unlabeled, addendum_size, model, BATCH)

        """
        X_initial, y_initial, X_unlabeled, y_unlabeled:初始标签集/初始非标签集分离出来的特征和标签(tensor)
        addendum_size:增加的数据量
        """
        # [0.7672, 0.7089, 0.8243, 0.9165, 0.9431, 0.8835, 0.9038, 0.9607, 0.963, 0.9538, 0.9722, 0.9911, 0.9728, 0.9664]

    if strategy == "MCD":
        X_initial, y_initial, X_unlabeled, y_unlabeled = MCD(X_initial, y_initial, X_unlabeled, y_unlabeled, addendum_size)
        # ↓可以封装一下
        new_train_dataset = TensorDataset(X_initial, y_initial.unsqueeze(1))
        train_loader = DataLoader(new_train_dataset, batch_size=BATCH, shuffle=True)
        # [0.7672, 0.6473, 0.7375, 0.5843, 0.723, 0.7502, 0.896, 0.9066, 0.9505, 0.955, 0.9462, 0.9573, 0.9631, 0.9582]



    if strategy == "GSx":
        pass

    if strategy== "EGAL":

        # 初始化EGAL采样器并进行采样
        sampler = EGAL(addendum_size=50, w=0.25)
        top_samples_indices = sampler.sample(X_train_full_df, labeled_indices, unlabeled_indices)

        # 更新 labeled_indices 和 unlabeled_indices
        labeled_indices.extend(top_samples_indices)
        unlabeled_indices = list(set(unlabeled_indices) - set(top_samples_indices))

        X_train_full_tensor = torch.tensor(X_train_full_df, dtype=torch.float32)
        y_train_full_tensor = torch.tensor(y_train_full_df, dtype=torch.float32)

        train_full_dataset = TensorDataset(X_train_full_tensor, y_train_full_tensor)

        labeled_subset = Subset(train_full_dataset, labeled_indices)

        # 创建 DataLoader
        train_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
        # [0.7672, 0.8275, 0.8218, 0.8464, 0.8817, 0.9029, 0.9193, 0.816, 0.8702, 0.8617, 0.8879, 0.9003, 0.8849, 0.9]

    if strategy == "BayesianAL":
        # 实例化贝叶斯主动学习类
        bal = BayesianAL(input_dim=NN_input, output_dim=NN_output, hid_dims=[64, 32], prior_scale=5.0)
        # 转换数据为Tensor
        X_train_labeled_tensor, y_train_labeled_tensor, X_train_unlabeled_tensor = bal.convert_to_tensor(X_train_full_df, y_train_full_df, labeled_indices, unlabeled_indices)

        # 训练模型
        bal.train(X_train_labeled_tensor, y_train_labeled_tensor)
        # 选择最不确定的样本
        selected_indices = bal.select_most_uncertain(X_train_unlabeled_tensor, unlabeled_indices, addendum_size)
        """
        selected_indices: list[int]
        包含在未标注数据集中，经过不确定性排序后选择的最不确定样本在原始训练数据集中的绝对索引。
        """
        # 更新 labeled_indices 和 unlabeled_indices
        labeled_indices.extend(selected_indices)
        for idx in selected_indices:
            unlabeled_indices.remove(idx)

        # 创建包含更新后的已标注样本子集
        labeled_subset = Subset(train_full_dataset, labeled_indices)
        # 创建 DataLoader
        train_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
        # [0.7672, 0.7934, 0.8716, 0.8794, 0.8361, 0.748, 0.7656, 0.7722, 0.8952, 0.9195, 0.9086, 0.8975, 0.9424, 0.931]









#MC dropout
#预期模型最大变化
#BMDAL
#L1 L2

# GS [0.7672, 0.8275, 0.8218, 0.8464, 0.8817, 0.9029, 0.9193, 0.816, 0.8702, 0.8617, 0.8879, 0.9003, 0.8849, 0.9]
# LL4AL [0.7672, 0.7411, 0.7061, 0.7707, 0.6642, 0.7191, 0.8511, 0.8372, 0.8936, 0.8973, 0.8771, 0.9368, 0.9316, 0.9551]
# BMDAL [0.7672, 0.7089, 0.8243, 0.9165, 0.9431, 0.8835, 0.9038, 0.9607, 0.963, 0.9538, 0.9722, 0.9911, 0.9728, 0.9664]
# MCD [0.7672, 0.6473, 0.7375, 0.5843, 0.723, 0.7502, 0.896, 0.9066, 0.9505, 0.955, 0.9462, 0.9573, 0.9631, 0.9582]
# EGAL [0.7672, 0.8275, 0.8218, 0.8464, 0.8817, 0.9029, 0.9193, 0.816, 0.8702, 0.8617, 0.8879, 0.9003, 0.8849, 0.9]
# BayesianAL [0.7672, 0.7934, 0.8716, 0.8794, 0.8361, 0.748, 0.7656, 0.7722, 0.8952, 0.9195, 0.9086, 0.8975, 0.9424, 0.931]