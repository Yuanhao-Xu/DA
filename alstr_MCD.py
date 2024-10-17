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
        self.train()  # Enable dropout for MC Dropout prediction
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
        Use MC Dropout for uncertainty query.

        Parameters:
        - X_train_labeled_df: Features of labeled training data.
        - y_train_labeled_df: Labels of labeled training data.
        - X_train_unlabeled_df: Features of unlabeled training data.
        - y_train_unlabeled_df: Labels of unlabeled training data.
        - addendum_size: Number of indices to select.
        - n_samples: Number of MC Dropout samples.
        - epochs: Number of training epochs.
        - lr: Learning rate.

        Returns:
        - selected_indices (list): List of selected unlabeled data indices.
        """
        # Automatically determine input and output dimensions
        input_dim = X_train_labeled_df.shape[1]
        output_dim = 1 if len(y_train_labeled_df.shape) == 1 else y_train_labeled_df.shape[1]

        # Check if model dimensions match
        if self.fc1.in_features != input_dim or self.fc3.out_features != output_dim:
            raise ValueError(f"Model input dimension should be {self.fc1.in_features}, output dimension should be {self.fc3.out_features}")

        # Convert DataFrame to tensors
        X_train_labeled_tensor = torch.tensor(X_train_labeled_df.values, dtype=torch.float32)
        y_train_labeled_tensor = torch.tensor(y_train_labeled_df.values, dtype=torch.float32)
        X_train_unlabeled_tensor = torch.tensor(X_train_unlabeled_df.values, dtype=torch.float32)

        # Train the model
        self.train_model(X_train_labeled_tensor, y_train_labeled_tensor, epochs=epochs, lr=lr)

        # MC Dropout prediction
        _, prediction_std = self.MCD_pred(X_train_unlabeled_tensor, n_samples=n_samples)

        # Sort by uncertainty and select top addendum_size samples
        uncertainty = prediction_std.flatten()
        incertitude_index = np.argsort(-uncertainty)[:addendum_size]

        return X_train_unlabeled_df.index[incertitude_index].tolist()
