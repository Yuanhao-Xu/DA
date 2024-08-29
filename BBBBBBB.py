import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from torch.nn import functional as F
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
import torch.nn as nn

class BayesianNN(PyroModule):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_dim, hid_dim)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hid_dim, in_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hid_dim]).to_event(1))

        self.fc2 = PyroModule[nn.Linear](hid_dim, out_dim)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([out_dim, hid_dim]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([out_dim]).to_event(1))

    def forward(self, x, y=None):
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze()

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(x, 1.).to_event(1), obs=y)
        return x


class BayesianAL:
    def __init__(self, X_train_labeled_df, y_train_labeled_df, X_train_unlabeled_df,
                 addendum_size, num_iterations=1000, lr=0.01):
        self.X_train_labeled_tensor = torch.tensor(X_train_labeled_df.values, dtype=torch.float32)
        self.y_train_labeled_tensor = torch.tensor(y_train_labeled_df.values, dtype=torch.float32)
        self.X_train_unlabeled_tensor = torch.tensor(X_train_unlabeled_df.values, dtype=torch.float32)

        self.addendum_size = addendum_size
        self.num_iterations = num_iterations

        self.X_train_unlabeled_df = X_train_unlabeled_df

        in_dim = self.X_train_labeled_tensor.shape[1]
        out_dim = self.y_train_labeled_tensor.shape[1]
        hid_dim = 64  # 可以根据需要调整隐藏层的维度

        # 初始化贝叶斯神经网络模型
        self.model = BayesianNN(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)
        self.guide = AutoDiagonalNormal(self.model)

        # 定义优化器
        self.optim = pyro.optim.Adam({"lr": lr})

        # 定义 SVI
        self.svi = SVI(self.model, self.guide, self.optim, loss=Trace_ELBO())

    def train(self):
        # 训练模型
        for j in range(self.num_iterations):
            loss = self.svi.step(self.X_train_labeled_tensor, self.y_train_labeled_tensor)
            if (j + 1) % 100 == 0:
                print(f"Iteration {j + 1:04d}: Loss = {loss:.4f}")

    def predict_with_uncertainty(self, X, num_samples=50):
        # 对模型进行多次采样，并进行预测
        sampled_models = [self.guide() for _ in range(num_samples)]
        yhats = []

        for model in sampled_models:
            # 使用采样的模型进行预测
            preds = self.model(X)
            yhats.append(preds.detach().numpy())

        # 将预测结果转换为 NumPy 数组
        yhats = np.array(yhats)  # 形状为 [num_samples, batch_size, out_dim]

        # 计算预测的均值和标准差
        mean = np.mean(yhats, axis=0)
        uncertainty = np.std(yhats, axis=0)

        return mean, uncertainty

    def query(self):
        # 训练模型
        self.train()

        # 获取不确定性
        _, predicted_stddevs = self.predict_with_uncertainty(self.X_train_unlabeled_tensor)


        # # 获取排序后的不确定性索引
        # uncertainty_indices = np.argsort(predicted_stddevs)[-self.addendum_size:]
        # incertitud
        #
        # return self.X_train_unlabeled_df.index[uncertainty_indices].tolist()
        # 获取排序后的不确定性索引（升序）
        sorted_indices = np.argsort(predicted_stddevs)

        # 选择最大的几个不确定性索引（降序排列）
        top_uncertainty_indices = sorted_indices[-self.addendum_size:]

        # 使用这些索引从未标记数据集中获取对应的行索引
        return self.X_train_unlabeled_df.index[top_uncertainty_indices].tolist()
