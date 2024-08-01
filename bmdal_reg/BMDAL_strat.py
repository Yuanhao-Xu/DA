# CreatTime 2024/7/31

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .bmdal.feature_data import TensorFeatureData
from .bmdal.algorithms import select_batch
# 传入参数 NNmodel 完整训练集

# 被选中的数据样本在原始池集中的位置

def BMDAL(X_initial, y_initial, X_unlabeled, y_unlabeled, addendum_size, custom_model, BATCH):

    train_data = TensorFeatureData(X_initial)
    pool_data = TensorFeatureData(X_unlabeled)
    new_idxs, _ = select_batch(batch_size=addendum_size, models=[custom_model],
                               data={'train': train_data, 'pool': pool_data}, y_train=y_initial,
                               selection_method='lcmd', sel_with_train=True,
                               base_kernel='grad', kernel_transforms=[('rp', [512])])

    # 将新选择的索引对应的数据加入到训练集中，并从池集中移除
    new_x_train = X_unlabeled[new_idxs]
    new_y_train = y_unlabeled[new_idxs].unsqueeze(1)

    X_initial = torch.cat((X_initial, new_x_train), dim=0)
    y_initial = torch.cat((y_initial.unsqueeze(1), new_y_train), dim=0).squeeze()  # 确保 y_initial 是零维张量

    # 删除池集中对应的样本
    mask = torch.ones(X_unlabeled.size(0), dtype=torch.bool)
    mask[new_idxs] = False

    X_unlabeled = X_unlabeled[mask]
    y_unlabeled = y_unlabeled[mask]
    new_train_dataset = TensorDataset(X_initial, y_initial.unsqueeze(1))

    return DataLoader(new_train_dataset, batch_size=BATCH, shuffle=True), X_initial, y_initial, X_unlabeled, y_unlabeled
