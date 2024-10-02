import random
import json
import os
from tqdm import tqdm
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import torch

# 导入自定义的 DataProcessor 模块
from BenchmarkModel import BenchmarkModel
from DataProcessor import *
from alstr_RS import RS
from alstr_LL4AL import LL4AL
from alstr_LCMD import LCMD
from alstr_MCD import MC_Dropout
from alstr_EGAL import EGAL
from alstr_BayesianAL import BayesianAL
from alstr_GSx import GSx
from alstr_GSy import GSy
from alstr_GSi import GSi
from alstr_GSBAG import GSBAG

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)

SEED = 50
set_seed(SEED)

# 定义策略
# strategies = ["RS"]
strategies = ["RS", "LL4AL", "LCMD", "MCD", "EGAL", "BayesianAL", "GSx", "GSy", "GSi", "GSBAG"]
addendum_size = 10
num_cycles = 85
NN_input = X_train_labeled_df.shape[1]
NN_output = y_train_labeled_df.shape[1]


# 字典保存每种策略的 test_R2s
R2s_dict = {}

# 实例化所有策略类
al_RS = RS()
al_LL4AL = LL4AL(BATCH=32, LR=0.001, MARGIN=0.7, WEIGHT=1.5, EPOCH=200, EPOCHL=30, WDECAY=5e-4)
al_LCMD = LCMD()
al_MCD = MC_Dropout(X_train_labeled_df.shape[1], 1)
al_EGAL = EGAL()
al_BayesianAL = BayesianAL()
al_GSx = GSx(random_state=42)
al_GSy = GSy(random_state=42)
al_GSi = GSi(random_state=42)
al_GSBAG = GSBAG(kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

# 遍历所有策略
for strategy in strategies:
    desc_text = f"[{strategy:^15}] ⇢ Cycles".ljust(10)

    # 初始化数据
    set_seed(SEED)
    current_labeled_indices = labeled_indices.copy()
    current_unlabeled_indices = unlabeled_indices.copy()
    current_X_train_labeled_df = X_train_labeled_df.copy()
    current_y_train_labeled_df = y_train_labeled_df.copy()
    current_X_train_unlabeled_df = X_train_unlabeled_df.copy()
    current_y_train_unlabeled_df = y_train_unlabeled_df.copy()

    test_R2s = []

    # 遍历主动学习的 cycle
    for cycle in tqdm(range(num_cycles), desc=f"{desc_text} ", ncols=80):

        # XGBoost 模型初始化，选择合适的参数
        model = xgb.XGBRegressor(
            n_estimators=1500,  # 迭代次数
            learning_rate=0.01,  # 学习率
            max_depth=6,  # 树的最大深度
            subsample=0.8,  # 子样本比例
            colsample_bytree=0.8,  # 每棵树使用特征的比例
            random_state=SEED
        )

        # 训练模型
        model.fit(current_X_train_labeled_df, current_y_train_labeled_df)

        # 测试模型
        y_pred = model.predict(X_test_df)
        test_R2 = round(r2_score(y_test_df, y_pred), 4)
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
            selected_indices = al_LCMD.query(BenchmarkModel(input_dim=NN_input, output_dim=NN_output),
                                             current_X_train_labeled_df,
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
                n_samples=50)

        elif strategy == "EGAL":
            selected_indices = al_EGAL.query(current_X_train_labeled_df,
                                             current_X_train_unlabeled_df,
                                             X_train_full_df, addendum_size,
                                             b_factor=0.1)

        elif strategy == "BayesianAL":
            selected_indices = al_BayesianAL.query(current_X_train_unlabeled_df, current_X_train_labeled_df,
                                                   current_y_train_labeled_df, addendum_size)

        elif strategy == "GSx":
            selected_indices = al_GSx.query(current_X_train_unlabeled_df, n_act=addendum_size)

        elif strategy == "GSy":
            selected_indices = al_GSy.query(current_X_train_unlabeled_df,
                                            addendum_size,
                                            current_X_train_labeled_df,
                                            current_y_train_labeled_df,
                                            current_y_train_unlabeled_df)

        elif strategy == "GSi":
            selected_indices = al_GSi.query(current_X_train_unlabeled_df,
                                            addendum_size,
                                            current_X_train_labeled_df,
                                            current_y_train_labeled_df,
                                            current_y_train_unlabeled_df)

        elif strategy == "GSBAG":
            al_GSBAG.fit(current_X_train_labeled_df, current_y_train_labeled_df)
            selected_indices = al_GSBAG.query(current_X_train_unlabeled_df,
                                              current_X_train_labeled_df,
                                              addendum_size)

        else:
            print("An undefined strategy was encountered.")  # 提示未定义的策略
            selected_indices = []

        # 更新索引
        current_labeled_indices.extend(selected_indices)
        for idx in selected_indices:
            current_unlabeled_indices.remove(idx)

        # 更新已标注和未标注数据集
        current_X_train_labeled_df = X_train_full_df.loc[current_labeled_indices]
        current_y_train_labeled_df = y_train_full_df.loc[current_labeled_indices]
        current_X_train_unlabeled_df = X_train_full_df.loc[current_unlabeled_indices]
        current_y_train_unlabeled_df = y_train_full_df.loc[current_unlabeled_indices]

    # 将每种策略的结果存入字典
    R2s_dict[strategy] = test_R2s

# 打印或保存所有策略的结果
print(R2s_dict)

# # 保存文件
# folder_name = 'xgb_res'
# if not os.path.exists(folder_name):
#     os.makedirs(folder_name)
#
# save_path = os.path.join(folder_name, 'ENB2012_1_10i_10s_60c_50s.json')
# with open(save_path, 'w') as f:
#     json.dump(R2s_dict, f)
#
# print(f"R2s_dict has been saved to {save_path}")
