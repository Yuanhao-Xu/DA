import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MC_Dropout(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1=64, hidden_dim2=32):
        super(MC_Dropout, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def MCD_pred(self, x_data, n_samples=50):
        self.train()  # 启用 dropout 进行 MC Dropout 预测
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                predictions.append(self.forward(x_data).cpu().numpy())
        predictions = np.array(predictions)
        prediction_mean = predictions.mean(axis=0)
        prediction_std = predictions.std(axis=0)
        return prediction_mean, prediction_std

    def train_model(self, x_train, y_train, epochs=500, lr=0.01):
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

    def query(self, X_train_labeled_df, y_train_labeled_df, X_train_unlabeled_df, y_train_unlabeled_df, addendum_size,
              n_samples=50, epochs=500, lr=0.01):
        """
        使用 MC Dropout 进行不确定性查询。

        参数:
        - X_train_labeled_df: 已标记训练数据的特征 DataFrame。
        - y_train_labeled_df: 已标记训练数据的标签 DataFrame。
        - X_train_unlabeled_df: 未标记训练数据的特征 DataFrame。
        - y_train_unlabeled_df: 未标记训练数据的标签 DataFrame。
        - addendum_size: 本次选择的索引数量。
        - n_samples: 进行 MC Dropout 预测时的采样次数。
        - epochs: 模型训练的轮数。
        - lr: 学习率。

        返回:
        - selected_indices (list): 本次选择的未标记数据的索引列表。
        """
        # 自动确定输入维度和输出维度
        input_dim = X_train_labeled_df.shape[1]
        output_dim = 1 if len(y_train_labeled_df.shape) == 1 else y_train_labeled_df.shape[1]

        # 检查模型的输入维度和输出维度是否匹配
        if self.fc1.in_features != input_dim or self.fc3.out_features != output_dim:
            raise ValueError(f"模型输入维度应为 {self.fc1.in_features}，输出维度应为 {self.fc3.out_features}")

        # 将 DataFrame 转换为张量
        X_train_labeled_tensor = torch.tensor(X_train_labeled_df.values, dtype=torch.float32)
        y_train_labeled_tensor = torch.tensor(y_train_labeled_df.values, dtype=torch.float32)
        X_train_unlabeled_tensor = torch.tensor(X_train_unlabeled_df.values, dtype=torch.float32)

        # 训练模型
        self.train_model(X_train_labeled_tensor, y_train_labeled_tensor, epochs=epochs, lr=lr)

        # 使用 MC Dropout 进行预测
        _, prediction_std = self.MCD_pred(X_train_unlabeled_tensor, n_samples=n_samples)

        # 对不确定性进行排序，选择不确定性最大的 addendum_size 个数据点
        uncertainty = prediction_std.flatten()
        incertitude_index = np.argsort(-uncertainty)[:addendum_size]

        return X_train_unlabeled_df.index[incertitude_index].tolist()
