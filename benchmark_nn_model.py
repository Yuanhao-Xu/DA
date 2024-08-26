# CreatTime 2024/7/25

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

class BenchmarkModel(nn.Module):
    def __init__(self, input_dim=8, output_dim=1):
        super(BenchmarkModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, self.output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ModelTrainer:
    def __init__(self, model, device=torch.device('cpu'), lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def train_model(self, train_loader, epochs=100):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)  # 确保在正确设备上
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)

            epoch_loss /= len(train_loader.dataset)
            # print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
        return self.model

    def evaluate_model(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)  # 确保在正确设备上
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                # 调用R2评估测试集
                r2 = r2_score(targets.cpu(), outputs.cpu())
                r2 = round(r2, 4)

        test_loss /= len(test_loader.dataset)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"r2_score: {r2}")

        return test_loss, r2
