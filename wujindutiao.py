import random
import torch
from torch.utils.data import Subset
import numpy as np
from torch.utils.data import DataLoader

# 导入自定义变量和函数
from DataProcessor import *
from benchmark_nn_model import BenchmarkModel, ModelTrainer
from alstr_RS import RS
from alstr_LL4AL import LL4AL
from alstr_LCMD import LCMD
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
# strategies = ["RS", "LL4AL", "LCMD", "MCD", "EGAL", "BayesianAL"]
strategies = ["BayesianAL"]

addendum_init = 100
BATCH = 32

num_cycles = 30
epochs = 500
addendum_size = 20

NN_input = X_train_labeled_df.shape[1]
NN_output = y_train_labeled_df.shape[1]

# 字典保存每种策略的 test_R2s
all_strategy_results = {}
# 实例化所有策略类
al_RS = RS()
al_LL4AL = LL4AL(BATCH=32, LR=0.001, MARGIN=0.7, WEIGHT=1.5, EPOCH=200, EPOCHL=30, WDECAY=5e-4)
al_LCMD = LCMD()
al_MCD = MC_Dropout(NN_input, NN_output)
al_EGAL = EGAL()
al_BayesianAL = BayesianAL()

# 遍历所有策略
for strategy in strategies:
    print(f"Executing strategy: {strategy}")

    # 初始化模型和数据
    set_seed(SEED)
    device = torch.device('cpu')
    model = BenchmarkModel(input_dim=NN_input, output_dim=NN_output)
    trainer = ModelTrainer(model, device=device, lr=0.001)

    # 初始化数据
    current_labeled_indices = labeled_indices.copy()
    current_unlabeled_indices = unlabeled_indices.copy()
    current_train_loader = train_loader
    current_X_train_labeled_df = X_train_labeled_df.copy()
    current_y_train_labeled_df = y_train_labeled_df.copy()
    current_X_train_unlabeled_df = X_train_unlabeled_df.copy()
    current_y_train_unlabeled_df = y_train_unlabeled_df.copy()

    test_R2s = []

    for cycle in range(num_cycles):

        print(f"Active Learning Cycle {cycle + 1}/{num_cycles} for {strategy}")

        # 训练模型
        model = trainer.train_model(current_train_loader, epochs)
        # 测试模型
        test_loss, test_R2 = trainer.evaluate_model(test_loader)

        test_R2s.append(test_R2)

        # 根据策略选择数据
        if strategy == "RS":
            selected_indices = al_RS.query(current_unlabeled_indices, addendum_size)

        elif strategy == "LL4AL":
            selected_indices = al_LL4AL.query(current_X_train_unlabeled_df,
                                              current_X_train_labeled_df,
                                              current_y_train_unlabeled_df,
                                              current_y_train_labeled_df,
                                              n_act=addendum_size)

        elif strategy == "LCMD":
            selected_indices = al_LCMD.query(model,current_X_train_labeled_df,
                                             current_y_train_labeled_df,
                                             current_X_train_unlabeled_df,
                                             current_y_train_unlabeled_df,
                                             addendum_size)

        elif strategy == "MCD":
            selected_indices = al_MCD.query(
                current_X_train_labeled_df,
                current_y_train_labeled_df,
                current_X_train_unlabeled_df,
                current_y_train_unlabeled_df,
                addendum_size=addendum_size,
                n_samples=50  # Monte Carlo采样次数
            )

        elif strategy == "EGAL":
            selected_indices = al_EGAL.query(current_X_train_labeled_df,
                                             current_X_train_unlabeled_df,
                                             X_train_full_df, addendum_size,
                                             b_factor = 0.1)

        elif strategy == "BayesianAL":
            selected_indices = al_BayesianAL.query(current_X_train_unlabeled_df, current_X_train_labeled_df,
                                                   current_y_train_labeled_df, addendum_size)

        else:
            print("An undefined strategy was encountered.")  # 提示未定义的策略
            selected_indices = []

        # 更新索引
        current_labeled_indices.extend(selected_indices)
        for idx in selected_indices:
            current_unlabeled_indices.remove(idx)
        # 创建包含更新后的已标注样本子集
        labeled_subset = Subset(train_full_dataset, current_labeled_indices)
        # 创建 DataLoader
        current_train_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
        # 索引更新后，更新数据
        current_X_train_labeled_df = X_train_full_df.loc[current_labeled_indices]
        current_y_train_labeled_df = y_train_full_df.loc[current_labeled_indices]
        current_X_train_unlabeled_df = X_train_full_df.loc[current_unlabeled_indices]
        current_y_train_unlabeled_df = y_train_full_df.loc[current_unlabeled_indices]

    # 将每种策略的结果存入字典
    all_strategy_results[strategy] = test_R2s

# 打印或保存所有策略的结果
print(all_strategy_results)