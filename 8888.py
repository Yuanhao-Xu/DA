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

# 定义主动学习策略列表
strategies = [
    "RS",
    "LL4AL",
    "LCMD",
    "MCD",
    "EGAL",
    "BayesianAL"
]

# 生成器函数，用于生成每次策略需要的数据副本
def data_generator():
    while True:
        yield (
            labeled_indices.copy(),
            unlabeled_indices.copy(),
            X_train_labeled_df.copy(),
            y_train_labeled_df.copy(),
            X_train_unlabeled_df.copy(),
            y_train_unlabeled_df.copy()
        )

data_gen = data_generator()

# 用于记录每种策略的test_R2s
strategy_test_R2s = {strategy: [] for strategy in strategies}

for strategy in strategies:
    # 获取当前策略的数据副本
    lbl_idx, ulb_idx, X_lbl, y_lbl, X_ulb, y_ulb = next(data_gen)

    for cycle in range(num_cycles):
        print(f"Active Learning Cycle {cycle + 1}/{num_cycles} for strategy {strategy}")

        # 训练模型
        model = trainer.train_model(train_loader, epochs)
        # 测试模型
        test_loss, test_R2 = trainer.evaluate_model(test_loader)

        strategy_test_R2s[strategy].append(test_R2)

        if strategy == "RS":
            selected_idx = RS(ulb_idx, addendum_size)

        elif strategy == "LL4AL":
            learning_loss = LL4AL(BATCH=32,
                                  LR=0.001,
                                  MARGIN=0.7,
                                  WEIGHT=1.5,
                                  EPOCH=200,
                                  EPOCHL=30,
                                  WDECAY=5e-4)
            selected_idx = learning_loss.query(X_ulb, X_lbl, y_ulb, y_lbl, n_act=addendum_size)

        elif strategy == "LCMD":
            selected_idx = LCMD(X_lbl, y_lbl, X_ulb, y_ulb, addendum_size, model, BATCH)

        elif strategy == "MCD":
            al_MCD = MC_Dropout(X_lbl, y_lbl)
            selected_idx = al_MCD.query(X_lbl, y_lbl, X_ulb, y_ulb, addendum_size=addendum_size, n_samples=50)

        elif strategy == "EGAL":
            al_EGAL = EGAL(X_lbl, X_ulb, X_train_full_df, addendum_size)
            selected_idx = al_EGAL.query()

        elif strategy == "BayesianAL":
            al_BayesianAL = BayesianAL()
            selected_idx = al_BayesianAL.query(X_ulb, X_lbl, y_lbl, addendum_size)

        lbl_idx.extend(selected_idx)
        for idx in selected_idx:
            ulb_idx.remove(idx)

        # 创建包含更新后的已标注样本子集
        labeled_subset = Subset(train_full_dataset, lbl_idx)
        # 创建 DataLoader
        train_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)

        # 索引更新后，传入参数也要更新
        X_lbl = X_train_full_df.loc[lbl_idx]
        y_lbl = y_train_full_df.loc[lbl_idx]
        X_ulb = X_train_full_df.loc[ulb_idx]
        y_ulb = y_train_full_df.loc[ulb_idx]

# 输出每种策略的test_R2s
for strategy, r2s in strategy_test_R2s.items():
    print(f"{strategy} Test R2 Scores: {r2s}")

#
# {'RS': [0.6266, 0.7221, 0.7251, 0.792, 0.7476, 0.6968, 0.8678, 0.8669, 0.8999, 0.9036, 0.9368, 0.9322, 0.9423, 0.9322],
#  'LL4AL': [0.926, 0.9524, 0.9329, 0.9092, 0.8646, 0.9438, 0.9315, 0.9515, 0.9553, 0.9331, 0.9501, 0.9594, 0.9487, 0.9585],
#  'LCMD': [0.9622, 0.9587, 0.9445, 0.9035, 0.8871, 0.8937, 0.9181, 0.9143, 0.9185, 0.9099, 0.9363, 0.9348, 0.9225, 0.936],
#  'MCD': [0.9427, 0.95, 0.9626, 0.95, 0.9044, 0.9265, 0.9419, 0.9489, 0.9285, 0.9152, 0.9251, 0.9205, 0.9152, 0.9334],
#  'EGAL': [0.9375, 0.9361, 0.9396, 0.9121, 0.9244, 0.9385, 0.8941, 0.9049, 0.9378, 0.921, 0.9294, 0.9133, 0.8945, 0.9224],
#  'BayesianAL': [0.8993, 0.9163, 0.902, 0.93, 0.9275, 0.9257, 0.9405, 0.9321, 0.8719, 0.8912, 0.879, 0.857, 0.866, 0.9071]}
