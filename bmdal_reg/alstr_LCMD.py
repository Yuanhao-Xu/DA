# CreatTime 2024/7/31

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .bmdal.feature_data import TensorFeatureData
from .bmdal.algorithms import select_batch


def LCMD(X_train_labeled_df, y_train_labeled_df, X_train_unlabeled_df, y_train_unlabeled_df, addendum_size, custom_model, BATCH):
    # 需要将df转化为tensor才能传入原方法
    X_train_labeled_tensor = torch.tensor(X_train_labeled_df.values, dtype=torch.float32)
    y_train_labeled_tensor = torch.tensor(y_train_labeled_df.values, dtype=torch.float32)
    X_train_unlabeled_tensor = torch.tensor(X_train_unlabeled_df.values, dtype=torch.float32)
    y_train_unlabeled_tensor = torch.tensor(y_train_unlabeled_df.values, dtype=torch.float32)

    train_data = TensorFeatureData(X_train_labeled_tensor)
    pool_data = TensorFeatureData(X_train_unlabeled_tensor)
    incertitude_index, _ = select_batch(batch_size=addendum_size, models=[custom_model],
                               data={'train': train_data, 'pool': pool_data}, y_train=y_train_labeled_tensor,
                               selection_method='lcmd', sel_with_train=True,
                               base_kernel='grad', kernel_transforms=[('rp', [512])])
    # 该方法返回的是未标记数据集的相对索引
    # Diplomarbeit/DA/bmdal_reg/bmdal/selection.py → print(f'Added {i+1} train samples to selection', flush=True)
    return X_train_unlabeled_df.index[incertitude_index].tolist()