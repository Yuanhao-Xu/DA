# Author: 
# CreatTime: 2024/8/7
# FileName：MCD_mian

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import os

from torch.utils.data import TensorDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

samples = 100
input = 8
hidden_1 = 64
hidden_2 = 32
output = 1

class DropoutNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(DropoutNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x.squeeze()
# 计算均值和方差，方差作为不确定度
def MCD_pred(model, x_data, n_samples=samples):
    model.train()  # 在预测时启用 dropout
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            predictions.append(model(x_data).cpu().numpy())
    predictions = np.array(predictions)
    prediction_mean = predictions.mean(axis=0)
    prediction_std = predictions.std(axis=0)
    return prediction_mean, prediction_std

def train_model(model, x_train, y_train, epochs=500, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

MCD_model = DropoutNN(input, hidden_1, hidden_2, output)

def MCD(x_train, y_train, x_pool, y_pool, addendum_size, n_samples=samples, model=MCD_model):

    train_model(model, x_train, y_train, epochs=500, lr=0.01)
    # 使用 MC Dropout 进行预测
    _, prediction_std = MCD_pred(model, x_pool, n_samples=n_samples)

    # 对不确定性进行排序，选择不确定性最大的 batch_size 个数据点
    uncertainty = prediction_std.flatten()
    query_indices = np.argsort(-uncertainty)[:addendum_size]

    query_x = x_pool[query_indices]
    query_y = y_pool[query_indices]

    # 将选择的数据点加入训练集
    x_train = torch.cat([x_train, query_x], dim=0)
    y_train = torch.cat([y_train, query_y], dim=0)

    # 使用布尔掩码从池中删除选择的数据点
    mask = torch.ones(len(x_pool), dtype=torch.bool)
    mask[query_indices] = False
    x_pool = x_pool[mask]
    y_pool = y_pool[mask]

    labeled_subset = TensorDataset(x_train, y_train)
    unlabeled_subset = TensorDataset(x_pool, y_pool.unsqueeze(1))

    return x_train, y_train, x_pool, y_pool

