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





# ==========参数==========
strategy = "RS"

ADDENDUM_init = 100
BATCH = 32

num_cycles = 14
epochs = 500


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
    print(f'Test Loss: {test_loss:.4f}')


    return test_loss,r2





### ====================================================主动学习过程
indices = list(range(len(X_train_full)))
random.shuffle(indices)
# 标签数据和无标签数据的索引号
labeled_set = indices[:ADDENDUM_init] # 初始数据集长度
unlabeled_set = indices[ADDENDUM_init:]

# 把整个训练集划分为标签子集和非标签子集
labeled_subset = Subset(train_full_dataset, labeled_set)
unlabeled_subset = Subset(train_full_dataset, unlabeled_set)

# 分离特征和标签
# X_initial, y_initial = labeled_subset
# X_unlabeled, y_unlabeled = unlabeled_subset

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
    """

    if strategy == "RS":

