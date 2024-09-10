import random
import torch
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from torch.utils.data import Subset
import numpy as np
from torch.utils.data import DataLoader

# 导入自定义变量和函数
from DataProcessor import *
from BenchmarkModel import BenchmarkModel, ModelTrainer
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
import pyro
from tqdm import tqdm
# 设置 Pyro 的随机种子
pyro.set_rng_seed(42)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)


SEED = 50 # 之前是50
set_seed(SEED)

# {'BayesianAL': [0.4501, 0.4517, 0.388], 'EGAL': [0.4501, 0.4911, 0.8158], 'LCMD': [0.4501, 0.5908, 0.7616],
#  'LL4AL': [0.4501, 0.5906, 0.8085], 'MCD': [0.4501, 0.442, 0.5853], 'RS': [0.4501, 0.668, 0.9289]}
# {'GSx': [0.4501, 0.8048, 0.934]}

# {'BayesianAL': [0.069, 0.2824, 0.4006],
#  'EGAL': [0.069, 0.2761, 0.7779],
#  'LCMD': [0.069, 0.4999, 0.821],
#  'LL4AL': [0.069, 0.1563, 0.2007],
#  'MCD': [0.069, 0.3795, 0.5281],
#  'RS': [0.069, 0.427, 0.792]}
#  非线性 r = 42
# {'BayesianAL': [0.3493, 0.4938, 0.675], 'EGAL': [0.3493, 0.5015, 0.893], 'LCMD': [0.3493, 0.7624, 0.8774],
#  'LL4AL': [0.3493, 0.516, 0.8887], 'MCD': [0.3493, 0.7501, 0.9267], 'RS': [0.3493, 0.5782, 0.915]}

# {'BayesianAL': [0.306, 0.3502, 0.3133],
#  'EGAL': [0.306, 0.4405, 0.4431],
#  'LCMD': [0.306, 0.5084, 0.4675],
#  'LL4AL': [0.306, 0.4252, 0.5097],
#  'MCD': [0.306, 0.3757, 0.4259],
#  'RS': [0.306, 0.5113, 0.5307]}

# {'RS': [0.306, 0.5113, 0.5307, 0.4975, 0.43, 0.4775],
#  'LL4AL': [0.306, 0.4252, 0.5097, 0.3957, 0.4778, 0.5262],
#  'LCMD': [0.306, 0.5084, 0.4675, 0.3835, 0.4362, 0.3658],
#  'MCD': [0.306, 0.3757, 0.4259, 0.6334, 0.5606, 0.5869],
#  'EGAL': [0.306, 0.4405, 0.4431, 0.43, 0.4932, 0.5097],
#  'BayesianAL': [0.306, 0.3502, 0.3133, 0.3525, 0.4431, 0.4154]}







# 基准模型参数
# strategies = ["RS", "LL4AL", "LCMD", "MCD", "EGAL", "BayesianAL", "GSx", "GSy", "GSi", "GSBAG"]
strategies = ["RS","LCMD"]
addendum_init = 100
addendum_size = 100
num_cycles = 7
epochs = 500
NN_input = X_train_labeled_df.shape[1]
NN_output = y_train_labeled_df.shape[1]

# 字典保存每种策略的 test_R2s
R2s_dict = {}
# 实例化所有策略类
al_RS = RS()
al_LL4AL = LL4AL(BATCH=32, LR=0.001, MARGIN=0.7, WEIGHT=1.5, EPOCH=200, EPOCHL=30, WDECAY=5e-4)
al_LCMD = LCMD()
al_MCD = MC_Dropout(NN_input, NN_output)
al_EGAL = EGAL()
al_BayesianAL = BayesianAL()
al_GSx = GSx(random_state=42)
al_GSy = GSy(random_state=42)
al_GSi = GSi(random_state=42)
al_GSBAG = GSBAG(kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

# 遍历所有策略
for strategy in strategies:
    desc_text = f"[{strategy:^15}] ⇢ Cycles".ljust(10)

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

    for cycle in tqdm(range(num_cycles), desc=f"{desc_text} ", ncols=80):

        # print(f"Active Learning Cycle {cycle + 1}/{num_cycles} for {strategy}")

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
                n_samples=50)

        elif strategy == "EGAL":
            selected_indices = al_EGAL.query(current_X_train_labeled_df,
                                             current_X_train_unlabeled_df,
                                             X_train_full_df, addendum_size,
                                             b_factor = 0.1)

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
    R2s_dict[strategy] = test_R2s

# 打印或保存所有策略的结果
print(R2s_dict)


# 初始数据集，采样点
# model based的模型和框架模型一致会怎么样 找一个更好的外层框架模型
# 外层模型改一下会怎么样
# 高斯过程的用高斯过程的外部框架
# 合成数据集sklearn 1000个数据以内    标签上+高斯噪声，哪一种抗噪能力强
#

# 1.数据集本身的问题
# 2.初始数据和采样数的选择
# 3.测哪些数据集
# 4.合成数据集
# 5.GSBAG


# 1.随即搜索设置为基线1
# 2.找特征维度数据集小一点的数据集
# 3.合成数据集，先不加噪声测试模型 sklearn
# 4.对比不同的边界条件 初始化数据方法，采样数量，初始数据集大小 100+10*n
