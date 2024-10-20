# Import necessary libraries
import random
import json
import os
from tqdm import tqdm
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import torch

# Import custom DataProcessor module
from LCMD_nn_Model import BenchmarkModel
from DataProcessor import *
from alstr_RS import RS
from alstr_LL4AL import LL4AL
from alstr_LCMD import LCMD
from alstr_MCD import MC_Dropout
from alstr_EGAL import EGAL
from alstr_BayesianAL import BayesianAL
from alstr_GSx import GSx
from alstr_GSy import GSy
from alstr_iGS import iGS
from alstr_GSBAG import GSBAG

# Set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU use
    np.random.seed(seed)
    random.seed(seed)

SEED = 50
set_seed(SEED)

# Define strategies
# strategies = ["iGS"]
strategies = ["RS", "LL4AL", "LCMD", "MCD", "EGAL", "BayesianAL", "GSx", "GSy", "iGS", "GSBAG"]
addendum_size = 10
num_cycles = 85
NN_input = X_train_labeled_df.shape[1]
NN_output = y_train_labeled_df.shape[1]

# Dictionary to store R² results for each strategy
R2s_dict = {}

# Initialize strategy classes
al_RS = RS()
al_LL4AL = LL4AL(BATCH=32, LR=0.001, MARGIN=0.7, WEIGHT=1.5, EPOCH=200, EPOCHL=30, WDECAY=5e-4)
al_LCMD = LCMD()
al_MCD = MC_Dropout(X_train_labeled_df.shape[1], 1)
al_EGAL = EGAL()
al_BayesianAL = BayesianAL()
al_GSx = GSx(random_state=42)
al_GSy = GSy(random_state=42)
al_iGS = iGS(random_state=42)
al_GSBAG = GSBAG(kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

# Loop through strategies
for strategy in strategies:
    desc_text = f"[{strategy:^15}] ⇢ Cycles".ljust(10)

    # Initialize data
    set_seed(SEED)
    current_labeled_indices = labeled_indices.copy()
    current_unlabeled_indices = unlabeled_indices.copy()
    current_X_train_labeled_df = X_train_labeled_df.copy()
    current_y_train_labeled_df = y_train_labeled_df.copy()
    current_X_train_unlabeled_df = X_train_unlabeled_df.copy()
    current_y_train_unlabeled_df = y_train_unlabeled_df.copy()

    test_R2s = []

    # Loop through active learning cycles
    for cycle in tqdm(range(num_cycles), desc=f"{desc_text} ", ncols=80):

        # XGBoost model initialization
        model = xgb.XGBRegressor(
            n_estimators=1500,  # Number of iterations
            learning_rate=0.01,  # Learning rate
            max_depth=6,  # Max tree depth
            subsample=0.8,  # Subsample ratio
            colsample_bytree=0.8,  # Feature ratio per tree
            random_state=SEED
        )

        # Train the model
        model.fit(current_X_train_labeled_df, current_y_train_labeled_df)

        # Test the model
        y_pred = model.predict(X_test_df)
        test_R2 = round(r2_score(y_test_df, y_pred), 4)
        test_R2s.append(test_R2)

        # Select data based on strategy
        if strategy == "RS":
            selected_indices = al_RS.query(current_unlabeled_indices, addendum_size)

        elif strategy == "LL4AL":
            selected_indices = al_LL4AL.query(current_X_train_unlabeled_df,
                                              current_X_train_labeled_df,
                                              current_y_train_unlabeled_df,
                                              current_y_train_labeled_df,
                                              n_act=addendum_size)

        elif strategy == "LCMD":
            selected_indices = al_LCMD.query(BenchmarkModel(input_dim=NN_input, output_dim=NN_output),
                                             current_X_train_labeled_df,
                                             current_y_train_labeled_df,
                                             current_X_train_unlabeled_df,
                                             current_y_train_unlabeled_df,
                                             addendum_size)

        elif strategy == "MCD":
            selected_indices = al_MCD.query(
                current_X_train_labeled_df,
                current_y_train_labeled_df,
                current_X_train_unlabeled_df,
                current_y_train_unlabeled_df,
                addendum_size=addendum_size,
                n_samples=50)

        elif strategy == "EGAL":
            selected_indices = al_EGAL.query(current_X_train_labeled_df,
                                             current_X_train_unlabeled_df,
                                             X_train_full_df, addendum_size,
                                             b_factor=0.15)

        elif strategy == "BayesianAL":
            selected_indices = al_BayesianAL.query(current_X_train_unlabeled_df, current_X_train_labeled_df,
                                                   current_y_train_labeled_df, addendum_size)

        elif strategy == "GSx":
            selected_indices = al_GSx.query(current_X_train_unlabeled_df, n_act=addendum_size)

        elif strategy == "GSy":
            selected_indices = al_GSy.query(current_X_train_unlabeled_df,
                                            addendum_size,
                                            current_X_train_labeled_df,
                                            current_y_train_labeled_df,
                                            current_y_train_unlabeled_df)

        elif strategy == "iGS":
            selected_indices = al_iGS.query(current_X_train_unlabeled_df,
                                            addendum_size,
                                            current_X_train_labeled_df,
                                            current_y_train_labeled_df,
                                            current_y_train_unlabeled_df)

        elif strategy == "GSBAG":
            al_GSBAG.fit(current_X_train_labeled_df, current_y_train_labeled_df)
            selected_indices = al_GSBAG.query(current_X_train_unlabeled_df,
                                              current_X_train_labeled_df,
                                              addendum_size)

        else:
            print("An undefined strategy was encountered.")  # Undefined strategy warning
            selected_indices = []

        # Update indices
        current_labeled_indices.extend(selected_indices)
        for idx in selected_indices:
            current_unlabeled_indices.remove(idx)

        # Update labeled and unlabeled datasets
        current_X_train_labeled_df = X_train_full_df.loc[current_labeled_indices]
        current_y_train_labeled_df = y_train_full_df.loc[current_labeled_indices]
        current_X_train_unlabeled_df = X_train_full_df.loc[current_unlabeled_indices]
        current_y_train_unlabeled_df = y_train_full_df.loc[current_unlabeled_indices]

    # Save R² results for each strategy
    R2s_dict[strategy] = test_R2s

# Print or save all strategy results
print(R2s_dict)

# Save file
# folder_name = 'xgb_res'
# if not os.path.exists(folder_name):
#     os.makedirs(folder_name)
#
# save_path = os.path.join(folder_name, 'GEN9f10n_10i_10s_85c_50s_20241015.json')
# with open(save_path, 'w') as f:
#     json.dump(R2s_dict, f)
#
# print(f"R2s_dict has been saved to {save_path}")
