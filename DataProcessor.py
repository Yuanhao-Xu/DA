import random

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, Subset

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed
set_seed(50)  # 42

class DataProcessor:
    def __init__(self, file_path, addendum_init, batch_size=32):
        self.file_path = file_path
        self.addendum_init = addendum_init
        self.batch_size = batch_size

        # Load data
        self.data = pd.read_csv(self.file_path)

        # Normalize data
        self.X = self.data.iloc[:, :-1].values  # Convert to NumPy array
        self.y = self.data.iloc[:, -1].values.reshape(-1, 1)  # Convert target to NumPy array (2D)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        self.X = scaler_X.fit_transform(self.X)
        self.y = scaler_y.fit_transform(self.y)

        # Convert to DataFrame with row/column labels
        self.X = pd.DataFrame(self.X, columns=self.data.columns[:-1], index=self.data.index)
        self.y = pd.DataFrame(self.y, columns=[self.data.columns[-1]], index=self.data.index)

        # Split into train and test sets
        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Reset index after splitting
        self.X_train_full = self.X_train_full.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_train_full = self.y_train_full.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)

        # Test set features/labels in DataFrame format
        self.X_test_df = self.X_test
        self.y_test_df = self.y_test

        # Get train set indices
        self.train_indices = self.X_train_full.index

        # Select labeled indices
        self.labeled_indices = np.random.choice(self.train_indices, size=self.addendum_init, replace=False).tolist()

        # Create unlabeled indices
        self.unlabeled_indices = [idx for idx in self.train_indices if idx not in self.labeled_indices]

        # Split into labeled/unlabeled sets
        self.X_train_labeled_df = self.X_train_full.loc[self.labeled_indices]
        self.y_train_labeled_df = self.y_train_full.loc[self.labeled_indices]

        self.X_train_unlabeled_df = self.X_train_full.loc[self.unlabeled_indices]
        self.y_train_unlabeled_df = self.y_train_full.loc[self.unlabeled_indices]

        # Convert full train/test sets to tensors
        self.X_train_full_tensor = torch.tensor(self.X_train_full.values, dtype=torch.float32)
        self.y_train_full_tensor = torch.tensor(self.y_train_full.values, dtype=torch.float32)

        self.X_test_tensor = torch.tensor(self.X_test.values, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(self.y_test.values, dtype=torch.float32)

        # Create test set loader
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Create train set loader
        self.train_full_dataset = TensorDataset(self.X_train_full_tensor, self.y_train_full_tensor)
        self.labeled_subset = Subset(self.train_full_dataset, self.labeled_indices)
        self.train_loader = DataLoader(self.labeled_subset, batch_size=self.batch_size, shuffle=True)


paths = {
    "UCI_concrete": "Dataset/concrete/concrete_data.csv",
    "BFRC_cs": "Dataset/BFRC/data_cs.csv",
    "BFRC_fs": "Dataset/BFRC/data_fs.csv",
    "BFRC_sts": "Dataset/BFRC/data_sts.csv",
    "Pullout_fmax": "Dataset/pullout/dataset_fmax.csv",
    "Pullout_ifss": "Dataset/pullout/dataset_ifss.csv",
    "CST": "Dataset/Concrete_Slump_Test/data.csv",
    "UHPC_cs": "Dataset/uhpc/Compressive_strength.csv",
    "UHPC_fs": "Dataset/uhpc/Flexural_strength.csv",
    "UHPC_mss": "Dataset/uhpc/Mini_slump_spread.csv",
    "UHPC_porosity": "Dataset/uhpc/Porosity.csv",
    "ENB2012_HL": "Dataset/ENB2012/data1.csv",
    "ENB2012_CL": "Dataset/ENB2012/data2.csv",
    "GEN3f5n": "G_Dataset/data_1100s_3f5n.csv",
    "GEN5f5n": "G_Dataset/data_1100s_5f5n.csv",
    "GEN7f5n": "G_Dataset/data_1100s_7f5n.csv",
    "GEN7f10n": "G_Dataset/data_1100s_7f10n.csv",
    "GEN7f15n": "G_Dataset/data_1100s_7f15n.csv",
    "GEN7f20n": "G_Dataset/data_1100s_7f20n.csv",
    "GEN9f5n": "G_Dataset/data_1100s_9f5n.csv",
    "GEN11f5n": "G_Dataset/data_1100s_11f5n.csv"
}

__all__ = [
    'train_loader',
    'test_loader',
    'labeled_indices',
    'unlabeled_indices',
    'train_full_dataset',
    'X_train_labeled_df',
    'y_train_labeled_df',
    'X_train_unlabeled_df',
    'y_train_unlabeled_df',
    'X_train_full_df',
    'y_train_full_df',
    'X_test_df',
    'y_test_df'
]

# Instantiate DataProcessor class
Dataset_UCI = DataProcessor(file_path=paths["GEN9f5n"], addendum_init=10)

# Get train loader
train_loader = Dataset_UCI.train_loader
test_loader = Dataset_UCI.test_loader

# Get labeled/unlabeled indices
labeled_indices = Dataset_UCI.labeled_indices
unlabeled_indices = Dataset_UCI.unlabeled_indices

# Get full train dataset
train_full_dataset = Dataset_UCI.train_full_dataset

# Get labeled/unlabeled datasets (features/labels)
X_train_labeled_df = Dataset_UCI.X_train_labeled_df
y_train_labeled_df = Dataset_UCI.y_train_labeled_df

X_train_unlabeled_df = Dataset_UCI.X_train_unlabeled_df
y_train_unlabeled_df = Dataset_UCI.y_train_unlabeled_df

# Get full train set (features/labels)
X_train_full_df = Dataset_UCI.X_train_full
y_train_full_df = Dataset_UCI.y_train_full

# For xgb
X_test_df = Dataset_UCI.X_test_df
y_test_df = Dataset_UCI.y_test_df
