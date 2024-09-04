import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import torch.nn as nn
from pyro.nn import PyroModule, PyroSample
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 定义贝叶斯神经网络
class BNN(PyroModule):
    def __init__(self, in_dim=8, out_dim=1, hid_dims=[64, 32], prior_scale=5.):
        super().__init__()

        self.activation = nn.ReLU()  # 使用ReLU激活函数
        assert in_dim > 0 and out_dim > 0 and len(hid_dims) > 0  # 确保维度有效

        # 定义层的大小列表：输入维度 + 隐藏层维度 + 输出维度
        self.layer_sizes = [in_dim] + hid_dims + [out_dim]

        # 单独定义层列表
        layer_list = [
            PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx])
            for idx in range(1, len(self.layer_sizes))
        ]

        # 将层列表转换为 PyroModule 的 ModuleList
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        # 为每一层的权重和偏置定义先验分布
        for idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(
                dist.Normal(0., prior_scale * np.sqrt(2 / self.layer_sizes[idx]))
                .expand([self.layer_sizes[idx + 1], self.layer_sizes[idx]])
                .to_event(2)
            )
            layer.bias = PyroSample(
                dist.Normal(0., prior_scale)
                .expand([self.layer_sizes[idx + 1]])
                .to_event(1)
            )

    def forward(self, x, y=None):
        x = x.reshape(-1, self.layer_sizes[0])  # 确保输入维度正确
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))  # 隐藏层之间的传递
        mu = self.layers[-1](x).squeeze(-1)  # 确保 mu 是 1D 向量 [batch_size]
        sigma = pyro.sample("sigma", dist.Gamma(2.0, 1.0))  # 生成标量 sigma

        # 广播 sigma 使其形状与 mu 一致
        sigma = sigma.expand_as(mu)

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)

        return mu

# 定义主动学习类
class BayesianAL:
    def __init__(self,
                 hid_dims=[64, 32],  # 默认隐藏层维度
                 prior_scale=5.0):  # 默认先验分布的尺度

        # 这里的输入维度和输出维度将由 query 方法中传入的数据决定
        self.hid_dims = hid_dims
        self.prior_scale = prior_scale

        # 初始化贝叶斯神经网络的其他组件，但不包括具体的数据
        self.guide = None
        self.svi = None

    def initialize_model(self, input_dim, output_dim):
        # 初始化贝叶斯神经网络和Pyro相关组件
        self.bnn = BNN(in_dim=input_dim, out_dim=output_dim, hid_dims=self.hid_dims, prior_scale=self.prior_scale)
        self.guide = pyro.infer.autoguide.AutoDiagonalNormal(self.bnn)
        self.optimizer = pyro.optim.Adam({"lr": 0.001})
        self.svi = pyro.infer.SVI(self.bnn, self.guide, self.optimizer, loss=pyro.infer.Trace_ELBO())

    def train(self, X_train_labeled_tensor, y_train_labeled_tensor, num_iterations=1500):
        for j in range(num_iterations):
            loss = self.svi.step(X_train_labeled_tensor, y_train_labeled_tensor)
            # if j % 500 == 0:
            #     print(f"Step {j} : loss = {loss}")

    def predict_with_uncertainty(self, X, num_samples=50):
        sampled_models = [self.guide() for _ in range(num_samples)]
        yhats = [self.bnn(X).detach().numpy() for model in sampled_models]
        mean = np.mean(yhats, axis=0)
        uncertainty = np.std(yhats, axis=0)
        return mean, uncertainty

    def query(self, X_train_unlabeled_df, X_train_labeled_df, y_train_labeled_df, addendum_size):
        # 根据输入数据的形状初始化模型
        input_dim = X_train_labeled_df.shape[1]
        output_dim = y_train_labeled_df.shape[1]
        self.initialize_model(input_dim, output_dim)

        # 将已标记和未标记的数据集转换为张量
        X_train_labeled_tensor = torch.tensor(X_train_labeled_df.values, dtype=torch.float32)
        y_train_labeled_tensor = torch.tensor(y_train_labeled_df.values, dtype=torch.float32)
        X_train_unlabeled_tensor = torch.tensor(X_train_unlabeled_df.values, dtype=torch.float32)

        # 训练模型
        self.train(X_train_labeled_tensor, y_train_labeled_tensor)

        # 计算不确定性并选择最不确定的样本索引
        _, uncertainties = self.predict_with_uncertainty(X_train_unlabeled_tensor)
        uncertainty_indices = np.argsort(uncertainties.flatten())[::-1][:addendum_size]
        selected_indices = X_train_unlabeled_df.index[uncertainty_indices].tolist()  # 从相对索引中取到绝对索引

        return selected_indices  # 返回绝对索引
