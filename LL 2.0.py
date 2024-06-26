# CreatTime 2024/6/26

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def set_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(50)

# 加载数据
file_path = 'Dataset/UCI_Concrete_Data.xls'
data = pd.read_excel(file_path)

# 数据标准化
X = data.iloc[:, :-1].values  # 特征变量
y = data.iloc[:, -1].values.reshape(-1, 1)  # 目标变量
# 实例化标准化器
scaler_X = StandardScaler()
scaler_y = StandardScaler()
# 标准化
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 初始分成训练集和测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_full_tensor = torch.tensor(X_train_full, dtype=torch.float32)
y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义神经网络
class ConcreteNet(nn.Module):
    def __init__(self):
        super(ConcreteNet, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        out1 = torch.relu(self.fc1(x))
        out2 = torch.relu(self.fc2(out1))
        out3 = torch.relu(self.fc3(out2))
        out = self.fc4(out3)
        return out, [out1, out2, out3]


# 定义LossNet，用于学习损失
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def set_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(50)

# 加载数据
file_path = 'Dataset/UCI_Concrete_Data.xls'
data = pd.read_excel(file_path)

# 数据标准化
X = data.iloc[:, :-1].values  # 特征变量
y = data.iloc[:, -1].values.reshape(-1, 1)  # 目标变量
# 实例化标准化器
scaler_X = StandardScaler()
scaler_y = StandardScaler()
# 标准化
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 初始分成训练集和测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_full_tensor = torch.tensor(X_train_full, dtype=torch.float32)
y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义神经网络
class ConcreteNet(nn.Module):
    def __init__(self):
        super(ConcreteNet, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        out1 = torch.relu(self.fc1(x))
        out2 = torch.relu(self.fc2(out1))
        out3 = torch.relu(self.fc3(out2))
        out = self.fc4(out3)
        return out, [out1, out2, out3]


# 定义LossNet，用于学习损失
class LossNet(nn.Module):
    def __init__(self):
        super(LossNet, self).__init__()
        self.fc1 = nn.Linear(112, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 自定义损失函数
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 确保在正确设备上
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
    return model


# 训练LossNet
def train_lossnet(lossnet, dataloader, criterion, optimizer, epochs=100):
    lossnet.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            _, features = models['backbone'](inputs)
            combined_features = torch.cat(features, dim=1)
            loss_outputs = lossnet(combined_features)
            target_loss = criterion(inputs, targets)
            loss = criterion(loss_outputs, target_loss)
            loss.backward()
            optimizer.step()
    return lossnet


# 使用LossNet进行主动学习选择
def select_samples_with_lossnet(model, lossnet, pool_loader, acquisition_size):
    model.eval()
    lossnet.eval()
    scores = []
    with torch.no_grad():
        for inputs, _ in pool_loader:
            inputs = inputs.to(device)
            _, features = model(inputs)
            combined_features = torch.cat(features, dim=1)
            loss_outputs = lossnet(combined_features)
            scores.extend(loss_outputs.cpu().numpy())

    scores = np.array(scores).flatten()
    selected_indices = np.argsort(scores)[-acquisition_size:]
    return selected_indices


# 测试模型
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 确保在正确设备上
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    return test_loss


# 计算精确度
def compute_accuracy(model, test_loader, scaler_y):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 确保在正确设备上
            outputs, _ = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())  # 使用 .extend 将每个批次的输出添加到列表中
            all_targets.extend(targets.cpu().numpy())  # 使用 .extend 将每个批次的目标添加到列表中

    # 反标准化
    all_predictions = scaler_y.inverse_transform(all_predictions)
    all_targets = scaler_y.inverse_transform(all_targets)

    accuracy = np.mean(1 - np.abs((all_predictions - all_targets) / all_targets)) * 100  # 计算百分比精确度
    return accuracy


# 主动学习训练过程
def active_learning_training(initial_data, full_data, model, lossnet, criterion, optimizer_model, optimizer_lossnet,
                             epochs=100, num_cycles=10, acquisition_size=50):
    X_pool, y_pool = full_data
    X_initial, y_initial = initial_data

    test_losses = []
    accuracies = []

    for cycle in range(num_cycles):
        print(f"Active Learning Cycle {cycle + 1}/{num_cycles}")

        # 构建训练数据集
        train_dataset = TensorDataset(X_initial, y_initial)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        pool_dataset = TensorDataset(X_pool, y_pool)
        pool_loader = DataLoader(pool_dataset, batch_size=32, shuffle=False)

        # 训练模型
        model = train_model(model, train_loader, criterion, optimizer_model, epochs)

        # 训练LossNet
        lossnet = train_lossnet(lossnet, train_loader, criterion, optimizer_lossnet, epochs)

        # 测试模型
        test_loss = evaluate_model(model, test_loader, criterion)
        test_losses.append(test_loss)

        # 计算精确度
        accuracy = compute_accuracy(model, test_loader, scaler_y)
        accuracies.append(accuracy)
        print(f'Cycle {cycle + 1}/{num_cycles}, Accuracy: {accuracy:.2f}%')

        # 使用LossNet选择新的数据点
        if len(X_pool) >= acquisition_size:
            selected_indices = select_samples_with_lossnet(model, lossnet, pool_loader, acquisition_size)
            X_new, y_new = X_pool[selected_indices], y_pool[selected_indices]

            # 移除被选中的数据点
            X_pool = np.delete(X_pool, selected_indices, axis=0)
            y_pool = np.delete(y_pool, selected_indices, axis=0)

            # 添加到初始训练集中
            X_initial = torch.cat((X_initial, torch.tensor(X_new, dtype=torch.float32).clone().detach()), dim=0)
            y_initial = torch.cat((y_initial, torch.tensor(y_new, dtype=torch.float32).clone().detach()), dim=0)
        else:
            print("Pool exhausted")
