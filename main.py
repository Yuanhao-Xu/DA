# CreatTime 2024/6/24
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# 加载数据
file_path = 'Dataset/UCI_Concrete_Data.xls'
data = pd.read_excel(file_path)

# 数据标准化
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values.reshape(-1, 1)  # 目标变量
# 实例
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
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
    return model


# 测试模型
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    return test_loss


# 主动学习训练过程
def active_learning_training(initial_data, full_data, model, criterion, optimizer, epochs=100, num_cycles=10,
                             acquisition_size=50):
    X_pool, y_pool = full_data
    X_initial, y_initial = initial_data

    test_losses = []

    for cycle in range(num_cycles):
        print(f"Active Learning Cycle {cycle + 1}/{num_cycles}")

        # 构建训练数据集
        train_dataset = TensorDataset(X_initial, y_initial)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 训练模型
        model = train_model(model, train_loader, criterion, optimizer, epochs)

        # 测试模型
        test_loss = evaluate_model(model, test_loader, criterion)
        test_losses.append(test_loss)

        # 随机采样新的数据点
        if len(X_pool) >= acquisition_size:
            indices = np.random.choice(len(X_pool), acquisition_size, replace=False)
            X_new, y_new = X_pool[indices], y_pool[indices]

            # 移除被选中的数据点
            X_pool = np.delete(X_pool, indices, axis=0)
            y_pool = np.delete(y_pool, indices, axis=0)

            # 添加到初始训练集中
            X_initial = torch.cat((X_initial, torch.tensor(X_new, dtype=torch.float32)), dim=0)
            y_initial = torch.cat((y_initial, torch.tensor(y_new, dtype=torch.float32)), dim=0)
        else:
            print("Pool exhausted")
            break

    # 可视化测试损失
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
    plt.xlabel('Active Learning Cycle')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.title('Active Learning Test Loss Over Cycles')
    plt.grid(True)
    plt.show()


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = ConcreteNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将初始训练数据分割为一小部分
initial_size = 50
X_initial, y_initial = X_train_full_tensor[:initial_size], y_train_full_tensor[:initial_size]
X_pool, y_pool = X_train_full_tensor[initial_size:], y_train_full_tensor[initial_size:]

# 执行主动学习过程
active_learning_training(
    initial_data=(X_initial, y_initial),
    full_data=(X_pool, y_pool),
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    epochs=20,
    num_cycles=100,
    acquisition_size=50
)
