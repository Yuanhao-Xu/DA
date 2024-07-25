# CreatTime 2024/7/25

import numpy as np
import torch


def RS(labeled_subset, unlabeled_subset, addendum_size):
    X_unlabeled, y_unlabeled = unlabeled_subset
    X_initial, y_initial = labeled_subset
    indices = np.random.choice(len(X_unlabeled), addendum_size, replace=False)

    labeled_set += indices
    unlabeled_set -= indices
