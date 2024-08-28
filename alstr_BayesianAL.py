import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal


class UpdatedBNN(PyroModule):
    def __init__(self, mu, stddev, in_dim=8, out_dim=1, hid_dim=5):
        super().__init__()
        self.mu = mu
        self.stddev = stddev

        self.activation = nn.Tanh()
        self.layer1 = PyroModule[nn.Linear](in_dim, hid_dim)
        self.layer2 = PyroModule[nn.Linear](hid_dim, out_dim)

        # 设置层1的权重和偏置
        self.layer1.weight = PyroSample(
            dist.Normal(self.mu[0:in_dim * hid_dim].reshape(hid_dim, in_dim),
                        self.stddev[0:in_dim * hid_dim].reshape(hid_dim, in_dim)).to_event(2)
        )
        self.layer1.bias = PyroSample(
            dist.Normal(self.mu[in_dim * hid_dim:in_dim * hid_dim + hid_dim],
                        self.stddev[in_dim * hid_dim:in_dim * hid_dim + hid_dim]).to_event(1)
        )

        # 设置层2的权重和偏置
        self.layer2.weight = PyroSample(
            dist.Normal(
                self.mu[in_dim * hid_dim + hid_dim:in_dim * hid_dim + hid_dim + hid_dim * out_dim].reshape(out_dim,
                                                                                                           hid_dim),
                self.stddev[in_dim * hid_dim + hid_dim:in_dim * hid_dim + hid_dim + hid_dim * out_dim].reshape(out_dim,
                                                                                                               hid_dim)).to_event(
                2)
        )
        self.layer2.bias = PyroSample(
            dist.Normal(self.mu[-out_dim:], self.stddev[-out_dim:]).to_event(1)
        )

    def forward(self, x, y=None):
        x = x.reshape(-1, x.shape[-1])
        x = self.activation(self.layer1(x))
        mu = self.layer2(x).squeeze(-1)

        # 通过 pyro.plate 指定我们处理批数据
        with pyro.plate("data", x.shape[0]):
            sigma = pyro.sample("sigma", dist.Gamma(1.0, 1.0))
            obs = pyro.sample("obs", dist.Normal(mu, sigma.expand(mu.shape)), obs=y)
        return mu



class BayesianAL:
    def __init__(self, X_train_labeled_df, y_train_labeled_df, X_train_unlabeled_df, y_train_unlabeled_df,
                 addendum_size=10, num_iterations=1000, lr=0.01):
        self.X_train_labeled_tensor = torch.tensor(X_train_labeled_df.values, dtype=torch.float32)
        self.y_train_labeled_tensor = torch.tensor(y_train_labeled_df.values, dtype=torch.float32)
        self.X_train_unlabeled_tensor = torch.tensor(X_train_unlabeled_df.values, dtype=torch.float32)

        self.addendum_size = addendum_size
        self.num_iterations = num_iterations

        self.X_train_unlabeled_df = X_train_unlabeled_df

        in_dim = self.X_train_labeled_tensor.shape[1]
        out_dim = self.y_train_labeled_tensor.shape[1]
        hid_dim = 5

        # 初始化模型参数
        mu = torch.zeros((in_dim * hid_dim) + (hid_dim * out_dim) + hid_dim + out_dim)
        stddev = torch.ones((in_dim * hid_dim) + (hid_dim * out_dim) + hid_dim + out_dim)

        # 定义贝叶斯神经网络模型
        self.model = UpdatedBNN(mu, stddev, in_dim=in_dim, out_dim=out_dim, hid_dim=hid_dim)
        self.guide = AutoDiagonalNormal(self.model)

        # 定义优化器
        self.optim = Adam({"lr": lr})

        # 定义 SVI
        self.svi = SVI(self.model, self.guide, self.optim, loss=Trace_ELBO())

    def train(self):
        # 训练模型
        for j in range(self.num_iterations):
            loss = self.svi.step(self.X_train_labeled_tensor, self.y_train_labeled_tensor)
            if (j + 1) % 100 == 0:
                print(f"Iteration {j + 1:04d}: Loss = {loss:.4f}")

    def predict(self):
        # 使用训练好的模型进行预测
        predictive = Predictive(self.model, guide=self.guide, num_samples=1000)
        preds = predictive(self.X_train_unlabeled_tensor)
        return preds['obs'].std(0)  # 返回标准差作为不确定性

    def query(self):
        # 训练模型
        self.train()

        # 获取不确定性
        predicted_stddevs = self.predict()

        # 获取排序后的不确定性索引（降序）
        uncertainty_indices = torch.argsort(predicted_stddevs, descending=True)[:self.addendum_size].tolist()

        return  self.X_train_unlabeled_df.index[uncertainty_indices]




