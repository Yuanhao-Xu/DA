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


class MC_Dropout(nn.Module):
    def __init__(self, X_train_labeled_df, y_train_labeled_df, hidden_dim1=64, hidden_dim2=32):
        super(MC_Dropout, self).__init__()

        # 自动确定 input_dim 和 output_dim
        input_dim = X_train_labeled_df.shape[1]
        output_dim = 1 if len(y_train_labeled_df.shape) == 1 else y_train_labeled_df.shape[1]

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
        return x

    def MCD_pred(self, x_data, n_samples=50):
        self.train()  # 在预测时启用 dropout
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                predictions.append(self.forward(x_data).cpu().numpy())
        predictions = np.array(predictions)
        prediction_mean = predictions.mean(axis=0)
        prediction_std = predictions.std(axis=0)
        return prediction_mean, prediction_std

    def train_model(self, x_train, y_train, epochs=500, lr=0.01):
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            # if epoch % 100 == 0:
            #     print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    def query(self, X_train_labeled_df, y_train_labeled_df, X_train_unlabeled_df, y_train_unlabeled_df, addendum_size,
              n_samples=50):
        X_train_labeled_tensor = torch.tensor(X_train_labeled_df.values, dtype=torch.float32)
        y_train_labeled_tensor = torch.tensor(y_train_labeled_df.values, dtype=torch.float32)
        X_train_unlabeled_tensor = torch.tensor(X_train_unlabeled_df.values, dtype=torch.float32)
        y_train_unlabeled_tensor = torch.tensor(y_train_unlabeled_df.values, dtype=torch.float32)

        self.train_model(X_train_labeled_tensor, y_train_labeled_tensor, epochs=500, lr=0.01)

        # 使用 MC Dropout 进行预测
        _, prediction_std = self.MCD_pred(X_train_unlabeled_tensor, n_samples=n_samples)

        # 对不确定性进行排序，选择不确定性最大的 addendum_size 个数据点
        uncertainty = prediction_std.flatten()
        incertitude_index = np.argsort(-uncertainty)[:addendum_size]

        return X_train_unlabeled_df.index[incertitude_index].tolist()
