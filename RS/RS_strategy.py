# CreatTime 2024/7/25

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

def RS(train_full_dataset, indices, addendum_size, ADDENDUM_init, BATCH, cycle):
    labeled_set = indices[:ADDENDUM_init+addendum_size*(cycle+1)]  # 初始数据集长度
    unlabeled_set = indices[ADDENDUM_init:]
    # 把整个训练集划分为标签子集和非标签子集
    labeled_subset = Subset(train_full_dataset, labeled_set)
    unlabeled_subset = Subset(train_full_dataset, unlabeled_set)
    return DataLoader(labeled_subset, batch_size=BATCH, shuffle=True)
