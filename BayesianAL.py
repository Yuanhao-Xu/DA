# Author: 
# CreatTime: 2024/8/20
# FileName：BNN
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
    def __init__(self, input_dim, output_dim, hid_dims, prior_scale):
        # 初始化贝叶斯神经网络和Pyro相关组件
        self.bnn = BNN(in_dim=input_dim, out_dim=output_dim, hid_dims=hid_dims, prior_scale=prior_scale)
        self.guide = pyro.infer.autoguide.AutoDiagonalNormal(self.bnn)
        self.optimizer = pyro.optim.Adam({"lr": 0.01})
        self.svi = pyro.infer.SVI(self.bnn, self.guide, self.optimizer, loss=pyro.infer.Trace_ELBO())

    def convert_to_tensor(self, X_train_full, y_train_full, labeled_indices, unlabeled_indices):
        """
        将训练数据转换为PyTorch的Tensor
        """
        X_train_labeled = torch.tensor(X_train_full[labeled_indices], dtype=torch.float32)
        y_train_labeled = torch.tensor(y_train_full[labeled_indices], dtype=torch.float32)
        X_train_unlabeled = torch.tensor(X_train_full[unlabeled_indices], dtype=torch.float32)
        return X_train_labeled, y_train_labeled, X_train_unlabeled

    def train(self, X_train, y_train, num_iterations=1500):
        for j in range(num_iterations):
            loss = self.svi.step(X_train, y_train)
            if j % 500 == 0:
                print(f"Step {j} : loss = {loss}")

    def predict_with_uncertainty(self, X, num_samples=50):
        sampled_models = [self.guide() for _ in range(num_samples)]
        yhats = [self.bnn(X).detach().numpy() for model in sampled_models]
        mean = np.mean(yhats, axis=0)
        uncertainty = np.std(yhats, axis=0)
        return mean, uncertainty

    def select_most_uncertain(self, X_train_unlabeled, unlabeled_indices, addendum_size):
        _, uncertainties = self.predict_with_uncertainty(X_train_unlabeled)
        uncertainty_indices = np.argsort(uncertainties.flatten())[::-1][:addendum_size]
        selected_indices = [unlabeled_indices[i] for i in uncertainty_indices]  # 从相对索引中取到绝对索引
        return selected_indices  # 取到的是绝对索引



if __name__ == "__main__":

    # 数据处理部分

    # 加载数据
    file_path = 'Dataset/CSV_UCI_Concrete_Data.csv'  # 请将路径替换为您的数据路径
    data = pd.read_csv(file_path)

    # 数据标准化
    X = data.iloc[:, :-1].values  # 输入特征
    y = data.iloc[:, -1].values.reshape(-1, 1)  # 目标变量

    # 标准化特征和目标变量
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # 分割数据为训练集和测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将训练集和测试集转化为DataFrame，以便获取潜在索引
    X_train_full_df = pd.DataFrame(X_train_full, columns=data.columns[:-1])
    X_test_df = pd.DataFrame(X_test, columns=data.columns[:-1])

    # 获取训练集和测试集的潜在索引
    train_indices = X_train_full_df.index.tolist()
    test_indices = X_test_df.index.tolist()

    # 随机选择初始标签集
    ADDENDUM_init = 100  # 设置初始标签集的大小
    labeled_indices = np.random.choice(train_indices, size=ADDENDUM_init, replace=False).tolist()

    # 剩余的训练集作为未标记数据集
    unlabeled_indices = list(set(train_indices) - set(labeled_indices))

    # 实例化贝叶斯主动学习类
    bal = BayesianAL(input_dim=X.shape[1], output_dim=1, hid_dims=[64, 32], prior_scale=5.0)

    # 转换数据为Tensor
    X_train_labeled, y_train_labeled, X_train_unlabeled = bal.convert_to_tensor(X_train_full, y_train_full, labeled_indices, unlabeled_indices)

    # 训练模型
    bal.train(X_train_labeled, y_train_labeled)

    # 选择最不确定的样本
    addendum_size = 10
    selected_indices = bal.select_most_uncertain(X_train_unlabeled, unlabeled_indices, addendum_size)

    print("Selected absolute indices:", selected_indices)