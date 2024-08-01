# CreatTime 2024/7/31

import torch
import torch.nn as nn

from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.algorithms import select_batch

from data_process import X_train_full_tensor, y_train_full_tensor

n_train = 100
n_pool = 2000
torch.manual_seed(1234)



x_train = X_train_full_tensor[:n_train]
y_train = y_train_full_tensor[:n_train]
x_pool = X_train_full_tensor[n_train:]
y_pool = y_train_full_tensor[n_train:]

custom_model = nn.Sequential(nn.Linear(8, 100), nn.SiLU(), nn.Linear(100, 100), nn.SiLU(), nn.Linear(100, 1))
opt = torch.optim.Adam(custom_model.parameters(), lr=2e-2)
for epoch in range(256):
    y_pred = custom_model(x_train)
    loss = ((y_pred - y_train)**2).mean()
    train_rmse = loss.sqrt().item()
    pool_rmse = ((custom_model(x_pool) - y_pool)**2).mean().sqrt().item()
    print(f'train RMSE: {train_rmse:5.3f}, pool RMSE: {pool_rmse:5.3f}')
    loss.backward()
    opt.step()
    opt.zero_grad()


train_data = TensorFeatureData(x_train)
pool_data = TensorFeatureData(x_pool)
new_idxs, _ = select_batch(batch_size=50, models=[custom_model],
                           data={'train': train_data, 'pool': pool_data}, y_train=y_train,
                           selection_method='lcmd', sel_with_train=True,
                           base_kernel='grad', kernel_transforms=[('rp', [512])])
