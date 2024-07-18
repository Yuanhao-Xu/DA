# CreatTime 2024/6/24
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

# 1.随机采样baseline 精确度出图
# 2.部署learn loss策略

plt.ion()
# 设置随机种子



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确保确定性行为
    torch.backends.cudnn.benchmark = False  # 确保可重复性


set_seed(50)

# 加载数据
file_path = 'Dataset/UCI_Concrete_Data.xls'
data = pd.read_excel(file_path)

# 数据标准化
X = data.iloc[:, :-1].values  # 转化np数组
y = data.iloc[:, -1].values.reshape(-1, 1)  # 目标变量
# 实例
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
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
            inputs, targets = inputs.to(device), targets.to(device)  # 确保在正确设备上
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
            inputs, targets = inputs.to(device), targets.to(device)  # 确保在正确设备上
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            # 调用R2评估测试集
            r2 = r2_score(targets, outputs)
            r2 = round(r2,4)

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')


    return test_loss,r2


# 计算精确度
def compute_accuracy(model, test_loader, scaler_y):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 确保在正确设备上
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 反标准化
    all_predictions = scaler_y.inverse_transform(all_predictions)
    all_targets = scaler_y.inverse_transform(all_targets)

    accuracy = np.mean(1 - np.abs((all_predictions - all_targets) / all_targets)) * 100  # 计算百分比精确度
    return accuracy


# 主动学习训练过程
def active_learning_training(initial_data, full_data, model, criterion, optimizer, epochs=100, num_cycles=10,
                             acquisition_size=50):
    X_pool, y_pool = full_data
    X_initial, y_initial = initial_data

    test_losses = []
    accuracies = []
    test_R2s = []

    for cycle in range(num_cycles):
        print(f"Active Learning Cycle {cycle + 1}/{num_cycles}")

        # 构建训练数据集
        train_dataset = TensorDataset(X_initial, y_initial)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 训练模型
        model = train_model(model, train_loader, criterion, optimizer, epochs)

        # 测试模型
        test_loss = evaluate_model(model, test_loader, criterion)[0]
        test_R2 = evaluate_model(model, test_loader, criterion)[1]
        test_losses.append(test_loss)
        test_R2s.append(test_R2)

        # 计算精确度
        accuracy = compute_accuracy(model, test_loader, scaler_y)
        accuracies.append(accuracy)
        print(f'Cycle {cycle + 1}/{num_cycles}, Accuracy: {accuracy:.2f}%')

        # 随机采样新的数据点
        if len(X_pool) >= acquisition_size:
            indices = np.random.choice(len(X_pool), acquisition_size, replace=False)
            X_new, y_new = X_pool[indices], y_pool[indices]

            # 移除被选中的数据点
            X_pool = np.delete(X_pool, indices, axis=0)
            y_pool = np.delete(y_pool, indices, axis=0)

            # 添加到初始训练集中
            X_initial = torch.cat((X_initial, torch.tensor(X_new, dtype=torch.float32).clone().detach()), dim=0)
            y_initial = torch.cat((y_initial, torch.tensor(y_new, dtype=torch.float32).clone().detach()), dim=0)
        else:
            print("Pool exhausted")
            break

    # 可视化测试损失和R2
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
    plt.xlabel('Active Learning Cycle')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.title('Active Learning Test Loss Over Cycles')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_R2s) + 1), accuracies, label='Accuracy')
    plt.xlabel('Active Learning Cycle')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Active Learning Accuracy Over Cycles')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    return test_R2s


# 设置设备为 CPU
device = torch.device('cpu')

# 初始化模型
model = ConcreteNet().to(device)  # 确保模型在 CPU 上
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 将初始训练数据分割为一小部分
initial_size = 100
X_initial, y_initial = X_train_full_tensor[:initial_size], y_train_full_tensor[:initial_size]
X_pool, y_pool = X_train_full_tensor[initial_size:], y_train_full_tensor[initial_size:]

# 执行主动学习过程


RS_R2_Score = active_learning_training(
    initial_data=(X_initial, y_initial),
    full_data=(X_pool, y_pool),
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    epochs=500,
    num_cycles=14,  # 为了测试方便，可以减少循环次数
    acquisition_size=50
)


# 使用训练好的模型预测测试集中的数据，并和测试集的target对比，生成一张测试集中所有预测值和真实值对比的图像
def plot_predictions(model, test_loader, scaler_y):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, targets_batch in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(targets_batch.numpy())

    # 反标准化
    predictions = scaler_y.inverse_transform(predictions)
    targets = scaler_y.inverse_transform(targets)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(targets, label='True Values')
    plt.plot(predictions, label='Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Concrete Compressive Strength')
    plt.legend()
    plt.title('Comparison of True and Predicted Values')
    plt.grid(True)
    plt.show()

# 预测并绘图
plot_predictions(model, test_loader, scaler_y)

# RS [0.582, 0.628, 0.774, 0.856, 0.902, 0.93, 0.92, 0.923, 0.928, 0.885, 0.904, 0.866, 0.932, 0.946]
# LL [0.3193, 0.5268, 0.6639, 0.7662, 0.8233, 0.8364, 0.8263, 0.8415, 0.8086, 0.8346, 0.8079, 0.8085, 0.8343, 0.8455]