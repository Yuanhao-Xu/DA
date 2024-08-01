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
from data_process import (X_train_full, X_test, y_train_full, y_test,
                          train_full_dataset, test_dataset, test_loader)
from pub_nnModel import ConcreteNet
from RS.RS_strat import RS
from LL4AL.LL_main_pro import LL4AL
from bmdal_reg.BMDAL_strat import BMDAL

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)

set_seed(50)

# ==========参数==========
strategy = "BMDAL"

ADDENDUM_init = 100
BATCH = 32

num_cycles = 14
epochs = 500
addendum_size = 50


# ==========深度学习参数==========

device = torch.device('cpu')
model = ConcreteNet().to(device)  # 确保模型在 CPU 上
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)



### 定义公用模型训练和测试过程

def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 确保在正确设备上
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        # print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
    return model

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 确保在正确设备上
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            # 调用R2评估测试集
            r2 = r2_score(targets, outputs)
            r2 = round(r2,4)

    test_loss /= len(test_loader.dataset)
    # print(f"Test Loss: {test_loss:.4f}")
    print(f"r2_score:{r2}")


    return test_loss,r2





### ====================================================主动学习过程
indices = list(range(len(X_train_full)))
random.shuffle(indices)
# 标签数据和无标签数据的索引号
labeled_set = indices[:ADDENDUM_init] # 初始数据集长度
unlabeled_set = indices[ADDENDUM_init:]

# # 把整个训练集划分为标签子集和非标签子集
# labeled_subset = Subset(train_full_dataset, labeled_set)
# unlabeled_subset = Subset(train_full_dataset, unlabeled_set)

# 定义分离特征和标签的函数
def split_features_labels(dataset, indices):
    subset = Subset(dataset, indices)
    features_list = []
    labels_list = []

    for i in range(len(subset)):
        features, label = subset[i]
        features_list.append(features)
        labels_list.append(label)

    features_tensor = torch.stack(features_list)
    labels_tensor = torch.tensor(labels_list)

    return subset, features_tensor, labels_tensor
labeled_subset, X_initial, y_initial = split_features_labels(train_full_dataset, labeled_set)
unlabeled_subset, X_unlabeled, y_unlabeled = split_features_labels(train_full_dataset, unlabeled_set)








# 建立标签子集的训练集
# train_dataset = TensorDataset(X_initial, y_initial)
train_loader = DataLoader(labeled_subset, batch_size=BATCH, shuffle=True)


test_losses = []
test_R2s = []

for cycle in range(num_cycles):

    print(f"Active Learning Cycle {cycle + 1}/{num_cycles}")

    # 训练模型
    model = train_model(model, train_loader, criterion, optimizer, epochs)
    # 测试模型
    test_loss = evaluate_model(model, test_loader, criterion)[0]
    test_R2 = evaluate_model(model, test_loader, criterion)[1]
    test_losses.append(test_loss)
    test_R2s.append(test_R2)
    """
    调用一种主动学习策略添加新的数据集索引，并更新标签集和非标签集
    方法：
    1.RS
    2.LL4AL
    3.BMDAL
    """

    if strategy == "RS":
        train_loader = RS(train_full_dataset, indices, addendum_size, ADDENDUM_init, BATCH, cycle)
        """
        train_full_dataset:完整训练集
        indices:完整训练集乱序的索引
        addendum_size:增加的数据量
        ADDENDUM_init:初始数据量
        BATCH:略
        cycle:循环数
        """
        # [0.7905, 0.8201, 0.8822, 0.8908, 0.9135, 0.9193, 0.9129, 0.9091, 0.9242, 0.8542, 0.8471, 0.9042, 0.9006, 0.9121]
    if strategy == "LL4AL":
        train_loader, unlabeled_subset = LL4AL(train_full_dataset, train_loader, labeled_set, unlabeled_set, unlabeled_subset, cycle)
        """
        train_full_dataset:完整训练集
        train_loader:初始训练加载器
        labeled_set:
        unlabeled_set:
        unlabeled_subset:
        cycle:
        """
        # [0.7672, 0.8395, 0.8732, 0.88, 0.8699, 0.8397, 0.8413, 0.8357, 0.8372, 0.8435, 0.8536, 0.8751, 0.8974, 0.9061]
    if strategy == "BMDAL":
        train_loader, X_initial, y_initial, X_unlabeled, y_unlabeled = BMDAL(X_initial, y_initial, X_unlabeled, y_unlabeled, addendum_size, model, BATCH)

        """
        X_initial, y_initial, X_unlabeled, y_unlabeled:初始标签集/初始非标签集分离出来的特征和标签(tensor)
        addendum_size:增加的数据量
        """
        # [0.7672, 0.7089, 0.8243, 0.9165, 0.9431, 0.8835, 0.9038, 0.9607, 0.963, 0.9538, 0.9722, 0.9911, 0.9728, 0.9664]