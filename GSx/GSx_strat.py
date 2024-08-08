# CreatTime 2024/8/8

# GSx，完全基于样本在特征空间中的位置进行选择的被动采样方法


import random
import sys

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.utils.random import check_random_state
from sklearn.ensemble import AdaBoostRegressor
import xgboost


class GSx():
    def __init__(self, X_pool, y_pool, labeled, budget, X_test, y_test):
        self.X_pool = X_pool # 样本
        self.y_pool = y_pool # 样本标签
        self.X_test = X_test # 测试集样本
        self.y_test = y_test # 测试集标签
        self.labeled = list(deepcopy(labeled)) # 已经标记的标签索引
        self.unlabeled = self.initialization() # 未标记的标签索引
        self.budgetLeft = deepcopy(budget) # 待标记的标签数量
        self.model = AdaBoostRegressor() # 利用模型对选出的数据结果进行测试

        self.MAEList = [] # 存储多次重复筛选的 MAE 结果，本代码只实现了一次，可在外层嵌套一层循环计算

    def D(self, a, b):
        return np.sqrt(sum((a - b) ** 2))

    def initialization(self):
        unlabeled = [i for i in range(len(self.y_pool))]
        for j in self.labeled:
            unlabeled.remove(j)
        return unlabeled

    def evaluation(self):
        """
        使用 K 折交叉验证，K 中的验证集作为测试集，取平均作为泛化误差
        :return:
        """
        # ------------------------------------- 筛选的数据进行验证 ------------------------------------

        X_train = self.X_pool[self.labeled]
        y_train = self.y_pool[self.labeled]

        # 在所有数据集上进行 K 折，将验证集作为测试集
        AL_Train_scores = []
        AL_Test_scores = []

        kfold = KFold(n_splits=10, shuffle=True).split(X_train, y_train)
        for k, (train, test) in enumerate(kfold):
            self.model.fit(X_train[train], y_train[train])

            AL_Train_score = mean_absolute_error(self.model.predict(X_train[train]), y_train[train])
            AL_Test_score = mean_absolute_error(self.model.predict(X_train[test]),y_train[test])

            AL_Train_scores.append(AL_Train_score)
            AL_Test_scores.append(AL_Test_score)

            # print('Fold: %2d, score: %.3f' % (k + 1, score))

        AL_Train_MAE = np.mean(AL_Train_scores)
        AL_Test_MAE = np.mean(AL_Test_scores)
        print('训练集 MAE：', AL_Train_MAE, '测试集 MAE：', AL_Test_MAE)

        return AL_Train_MAE, AL_Test_MAE

    def select(self):
        while self.budgetLeft > 0:
            max_metric = OrderedDict() # 定义有序字典
            for idx in self.unlabeled: # 遍历未被标记的样本
                little = np.inf
                for jdx in self.labeled: # 遍历所有标记的样本
                    dist = self.D(self.X_pool[idx], self.X_pool[jdx]) # 计算未标记样本 idx 已标记样本 jdx 的欧式距离
                    if dist < little: # 选择最小的距离
                        little = dist
                max_metric[idx] = little # 记录未标记样本 idx 与所有标记样本的最短距离
            tar_idx = max(max_metric, key=max_metric.get) # 选择最短距离最大的样本
            self.labeled.append(tar_idx) # 将其标记，添加到标签中
            self.unlabeled.remove(tar_idx) # 将其在未标记的标签中移除
            self.budgetLeft -= 1

# ---------------------------------------- 产生数据集 --------------------------------------
# Training samples
X_train = np.random.uniform(-1, 1, 1000).reshape(500, 2)
y_train = X_train[:, 0]**2 - X_train[:, 1]**2 + X_train[:, 1] - 1

# Testing samples
X_test = np.random.uniform(-1, 1, 100).reshape(50, 2)
y_test = X_test[:, 0]**2 - X_test[:, 1]**2 + X_test[:, 1] - 1

# ----------------------------------------- 在所有数据集上进行训练评估 ----------------------------
model = AdaBoostRegressor()

# ----------------------------------------- K 折交叉验证，验证模型的性能 ------------------------
kfold = KFold(n_splits=10, shuffle=True).split(X_train, y_train)

# 在所有数据集上进行 K 折，将验证集作为测试集
Train_scores = []
Test_scores = []
for k, (train, test) in enumerate(kfold):

    model.fit(X_train[train], y_train[train])

    Train_score = mean_absolute_error(model.predict(X_train[train]), y_train[train])
    Test_score = mean_absolute_error(model.predict(X_train[test]), y_train[test])

    Train_scores.append(Train_score)
    Test_scores.append(Test_score)

    # print('Fold: %2d, score: %.3f' % (k + 1, score))

Train_MAE = np.mean(Train_scores)
Test_MAE = np.mean(Test_scores)
print('训练集 MAE：', Train_MAE, '测试集 MAE：', Test_MAE)

# --------------------------------------------------- 进行样本选择 ------------------------------------------------
labeled = [6, 16, 26, 36] # 未标记的数据索引
Budget = 100 # 需要标记的标签个数

ALmodel = GSx(X_pool=X_train, y_pool=y_train, labeled=labeled, budget=Budget, X_test=X_test, y_test=y_test)
ALmodel.select()
ALmodel.evaluation()
