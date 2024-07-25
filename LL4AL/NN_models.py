# Author: 
# CreatTime: 2024/7/24
# FileName：NN_models
import random


import torch
import torch.nn as nn
import torch.nn.functional as F


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.relu(self.fc3(x2))
        x = self.fc4(x3)
        return x,[x1,x2,x3] # 返回预测结果和中间层特征


class LossNet(nn.Module):
    def __init__(self, interm_dim=16):
        super(LossNet, self).__init__()
        # 定义每个中间特征的全连接层
        self.fc1 = nn.Linear(64, interm_dim)
        self.fc2 = nn.Linear(32, interm_dim)
        self.fc3 = nn.Linear(16, interm_dim)

        # 最终全连接层，将拼接后的特征映射到一个标量值
        self.linear = nn.Linear(3 * interm_dim, 1)

    def forward(self, features):
        # 提取每个中间特征并通过全连接层
        x1 = F.relu(self.fc1(features[0]))
        x2 = F.relu(self.fc2(features[1]))
        x3 = F.relu(self.fc3(features[2]))

        # 拼接所有特征
        x = torch.cat((x1, x2, x3), dim=1)

        # 通过最终全连接层，得到损失预测值
        x = self.linear(x)
        return x

