# CreatTime 2024/7/31

import torch
import torch.nn as nn

from bmdal.feature_data import TensorFeatureData
from bmdal.algorithms import select_batch
# 传入参数 NNmodel 完整训练集

# 被选中的数据样本在原始池集中的位置

def BMDAL(X_train_full_tensor, y_train_full_tensor, addendum_init, addendum_size, custom_model):
    x_train = X_train_full_tensor[:addendum_init]
    y_train = y_train_full_tensor[:addendum_init]
    x_pool = X_train_full_tensor[addendum_init:]
    y_pool = y_train_full_tensor[addendum_init:]

    train_data = TensorFeatureData(x_train)
    pool_data = TensorFeatureData(x_pool)
    new_idxs, _ = select_batch(batch_size=50, models=[custom_model],
                               data={'train': train_data, 'pool': pool_data}, y_train=y_train,
                               selection_method='lcmd', sel_with_train=True,
                               base_kernel='grad', kernel_transforms=[('rp', [512])])
