# CreatTime 2024/6/26
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import visdom
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import r2_score


class MainNet(nn.Module):
    def __init__(self, input_dim):
        super(MainNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.relu(self.fc3(x2))
        x = self.fc4(x3)
        return x, [x1, x2, x3]  # Return prediction and intermediate features


class LossNet(nn.Module):
    def __init__(self, interm_dim=16):
        super(LossNet, self).__init__()
        # Define fully connected layers for each intermediate feature
        self.fc1 = nn.Linear(64, interm_dim)
        self.fc2 = nn.Linear(32, interm_dim)
        self.fc3 = nn.Linear(16, interm_dim)

        # Final fully connected layer to map concatenated features to a scalar
        self.linear = nn.Linear(3 * interm_dim, 1)

    def forward(self, features):
        # Pass each intermediate feature through a fully connected layer
        x1 = F.relu(self.fc1(features[0]))
        x2 = F.relu(self.fc2(features[1]))
        x3 = F.relu(self.fc3(features[2]))

        # Concatenate all features
        x = torch.cat((x1, x2, x3), dim=1)

        # Pass through the final fully connected layer
        x = self.linear(x)
        return x


class LL4AL:
    def __init__(self, BATCH=32, LR=0.001, MARGIN=0.7, WEIGHT=1.5, EPOCH=200, EPOCHL=30, WDECAY=5e-4):
        self.BATCH = BATCH
        self.LR = LR
        self.MARGIN = MARGIN
        self.WEIGHT = WEIGHT
        self.EPOCH = EPOCH
        self.EPOCHL = EPOCHL
        self.WDECAY = WDECAY
        self.iters = 0
        self.device = torch.device('cpu')

    # ============================== Active Learning Functions ==============================
    def LossPredLoss(self, input, target, margin=1.0, reduction='mean'):
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape

        input = (input - input.flip(0))[:len(input) // 2]
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()

        # Loss function focuses on ranking the losses rather than exact values
        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1
        loss = None
        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0)
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss

    # Compute loss on unlabeled data for uncertainty estimation
    def get_uncertainty(self, models, unlabeled_loader, criterion):
        models['backbone'].eval()
        models['module'].eval()
        uncertainty = torch.tensor([])

        with torch.no_grad():
            for (inputs, labels) in unlabeled_loader:
                inputs = inputs

                scores, features = models['backbone'](inputs)
                pred_loss = models['module'](features)

                pred_loss = pred_loss.view(pred_loss.size(0))

                uncertainty = torch.cat((uncertainty, pred_loss), 0)

        return uncertainty.cpu()

    # Train utilities
    def train_epoch(self, models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
        models['backbone'].train()
        models['module'].train()

        for data in dataloaders['train']:
            inputs = data[0]
            labels = data[1]

            optimizers['backbone'].zero_grad()
            optimizers['module'].zero_grad()

            scores, features = models['backbone'](inputs)
            target_loss = criterion(scores, labels)

            if epoch > epoch_loss:
                # Stop gradient backpropagation from loss prediction module after certain epochs
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss = self.LossPredLoss(pred_loss, target_loss, margin=self.MARGIN)
            loss = m_backbone_loss + self.WEIGHT * m_module_loss

            loss.backward()
            optimizers['backbone'].step()
            optimizers['module'].step()

    # Modified for regression task (original paper was classification)
    def test(self, models, dataloaders, criterion, mode='val'):
        assert mode == 'val' or mode == 'test'
        models['backbone'].eval()
        models['module'].eval()

        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloaders['test']:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, features = models['backbone'](inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        test_loss /= len(dataloaders['test'].dataset)
        print(f'Test Loss: {test_loss:.4f}')

        # Evaluate with R2 score
        r2 = r2_score(targets, outputs)
        r2 = round(r2, 4)
        print(f"r2_score : {r2:.4f}")

        return test_loss, r2

    def train(self, models, criterion, optimizers, dataloaders, num_epochs, epoch_loss):
        for epoch in range(num_epochs):
            self.train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss)

    # ============================== Main Function ==============================
    def query(self, X_unlabeled, X_labeled, y_unlabeled, y_labeled, n_act=1):
        """
        :param X_unlabeled: Features of unlabeled dataset (dataframe)
        :param X_labeled: Features of labeled dataset (dataframe)
        :param y_unlabeled: Targets of unlabeled dataset (dataframe)
        :param y_labeled: Targets of labeled dataset (dataframe)
        :return: Suggested indices
        """
        # Create dataloaders for labeled and unlabeled datasets
        X_labeled_tensor = torch.tensor(X_labeled.values).float()
        y_labeled_tensor = torch.tensor(y_labeled.values).float()
        X_unlabeled_tensor = torch.tensor(X_unlabeled.values).float()
        y_unlabeled_tensor = torch.tensor(y_unlabeled.values).float()

        train_loader = DataLoader(TensorDataset(X_labeled_tensor, y_labeled_tensor), batch_size=self.BATCH, shuffle=True)
        unlabeled_loader = DataLoader(TensorDataset(X_unlabeled_tensor, y_unlabeled_tensor), batch_size=self.BATCH, shuffle=False)
        dataloaders = {'train': train_loader}

        # Model
        mainnet = MainNet(X_labeled.shape[1]).to(self.device)
        lossnet = LossNet(X_unlabeled.shape[1]).to(self.device)
        models = {'backbone': mainnet, 'module': lossnet}

        criterion_train = nn.MSELoss(reduction='none')  # Calculate loss for each sample
        optim_backbone = optim.AdamW(models['backbone'].parameters(), lr=self.LR, weight_decay=self.WDECAY)
        optim_module = optim.AdamW(models['module'].parameters(), lr=self.LR, weight_decay=self.WDECAY)

        optimizers = {'backbone': optim_backbone, 'module': optim_module}

        # Training
        self.train(models, criterion_train, optimizers, dataloaders, self.EPOCH, self.EPOCHL)

        # Measure uncertainty for unlabeled data
        uncertainty = self.get_uncertainty(models, unlabeled_loader, criterion_train)

        # Convert uncertainty to numpy
        uncertainty_numpy = uncertainty.numpy()
        # Select indices of the top 'n_act' uncertain samples
        incertitude_index = np.argsort(uncertainty_numpy)[-n_act:]

        return X_unlabeled.index[incertitude_index].tolist()
