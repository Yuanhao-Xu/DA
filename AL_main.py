# CreatTime 2024/7/25
# 导入库
import random
import torch
from torch.utils.data import Subset
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

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
from alstr_RS import RS
from alstr_LL4AL import LL4AL
from bmdal_reg.alstr_LCMD import LCMD
from alstr_MCD import MC_Dropout
from alstr_EGAL import EGAL
from alstr_BayesianAL import BayesianAL
from alstr_GSx import GSx
from alstr_GSy import GSy
from alstr_GSi import GSi
from alstr_GSBAG import GSBAG
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
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
strategy = "GSBAG"

addendum_init = 100
BATCH = 32

num_cycles = 30
epochs = 500
addendum_size = 20

NN_input = 8
NN_output = 1

# 基准模型初始化

device = torch.device('cpu')
model = BenchmarkModel(input_dim=NN_input, output_dim=NN_output)
trainer = ModelTrainer(model, device=device, lr=0.001)



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
    3.LCMD
    4.MCD
    5.EGAL
    6.BayesianAL
    7.GSx
    8.GSy
    9.GSi
    10.GSBAG
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

    if strategy == "LL4AL":
        learning_loss = LL4AL(BATCH=32,
                              LR=0.001,
                              MARGIN=0.7,
                              WEIGHT=1.5,
                              EPOCH=200,
                              EPOCHL=30,
                              WDECAY=5e-4)
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
        # [0.643, 0.7663, 0.7391, 0.7832, 0.8305, 0.7676, 0.8224, 0.9093, 0.9078, 0.8851, 0.8703, 0.8665, 0.9004, 0.9067]



    if strategy == "LCMD":
        selected_indices = LCMD(X_train_labeled_df,
                                y_train_labeled_df,
                                X_train_unlabeled_df,
                                y_train_unlabeled_df,
                                addendum_size, model,
                                BATCH)

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

        # [0.7672, 0.7089, 0.8243, 0.9165, 0.9431, 0.8835, 0.9038, 0.9607, 0.963, 0.9538, 0.9722, 0.9911, 0.9728, 0.9664]
        # [0.5176, 0.792, 0.8802, 0.8883, 0.9074, 0.8811, 0.8996, 0.8981, 0.9537, 0.946, 0.9544, 0.9528, 0.9775, 0.9716]






    if strategy == "MCD":
        al_MCD = MC_Dropout(X_train_labeled_df, y_train_labeled_df)
        selected_indices = al_MCD.query(
            X_train_labeled_df,
            y_train_labeled_df,
            X_train_unlabeled_df,
            y_train_unlabeled_df,
            addendum_size=addendum_size,
            n_samples=50  # Monte Carlo采样次数
        )

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

        # [0.7672, 0.6473, 0.7375, 0.5843, 0.723, 0.7502, 0.896, 0.9066, 0.9505, 0.955, 0.9462, 0.9573, 0.9631, 0.9582]

    if strategy== "EGAL":
        al_EGAL = EGAL(X_train_labeled_df, X_train_unlabeled_df, X_train_full_df, addendum_size)
        selected_indices = al_EGAL.query()
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
        # [0.6266, 0.9119, 0.9055, 0.8543, 0.8638, 0.9228, 0.9405, 0.9475, 0.8938, 0.8818, 0.8233, 0.8395, 0.8921, 0.8966]


    if strategy == "BayesianAL":
        al_BayesianAL = BayesianAL()
        selected_indices = al_BayesianAL.query(X_train_unlabeled_df, X_train_labeled_df, y_train_labeled_df, addendum_size)

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

    # if strategy == "GSx":
    #     al_GSx = GSx(random_state=42)
    #     # 使用 query 方法选择 2 个样本
    #     selected_indices = al_GSx.query(X_train_unlabeled_df, n_act=addendum_size)
    #     labeled_indices.extend(selected_indices)
    #     for idx in selected_indices:
    #         unlabeled_indices.remove(idx)
    #     # 创建包含更新后的已标注样本子集
    #     labeled_subset = Subset(train_full_dataset, labeled_indices)
    #     # 创建 DataLoader
    #     train_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
    #     # 索引更新后，传入参数也要更新
    #     X_train_labeled_df = X_train_full_df.loc[labeled_indices]
    #     y_train_labeled_df = y_train_full_df.loc[labeled_indices]
    #     X_train_unlabeled_df = X_train_full_df.loc[unlabeled_indices]
    #     y_train_unlabeled_df = y_train_full_df.loc[unlabeled_indices]
    #
    # if strategy == "GSy":
    #     al_GSy = GSy(random_state=42)
    #     # 使用 query 方法选择 2 个样本
    #     selected_indices = al_GSy.query(X_train_unlabeled_df, addendum_size, X_train_labeled_df, y_train_labeled_df, y_train_unlabeled_df)
    #     labeled_indices.extend(selected_indices)
    #     for idx in selected_indices:
    #         unlabeled_indices.remove(idx)
    #     # 创建包含更新后的已标注样本子集
    #     labeled_subset = Subset(train_full_dataset, labeled_indices)
    #     # 创建 DataLoader
    #     train_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
    #     # 索引更新后，传入参数也要更新
    #     X_train_labeled_df = X_train_full_df.loc[labeled_indices]
    #     y_train_labeled_df = y_train_full_df.loc[labeled_indices]
    #     X_train_unlabeled_df = X_train_full_df.loc[unlabeled_indices]
    #     y_train_unlabeled_df = y_train_full_df.loc[unlabeled_indices]
    #
    # if strategy == "GSi":
    #     al_GSi = GSi(random_state=42)
    #     selected_indices = al_GSi.query(X_train_unlabeled_df, addendum_size, X_train_labeled_df, y_train_labeled_df, y_train_unlabeled_df)
    #
    # if strategy == "GSBAG":
    #     kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
    #     al_GSBAG = GSBAG(random_state=42, kernel=kernel)
    #     selected_indices = al_GSBAG.query(X_train_unlabeled_df, X_train_labeled_df, addendum_size)













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