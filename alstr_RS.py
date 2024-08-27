# CreatTime 2024/7/25

import numpy as np

def RS(unlabeled_indices, addendum_size):
    """
    从未标记索引中随机选择 addendum_size 个索引。

    参数:
    - unlabeled_indices (list): 当前未标记数据的索引列表。
    - addendum_size (int): 每次选择的索引数量。

    返回:
    - selected_indices (list): 本次选择的索引列表。
    """
    if len(unlabeled_indices) < addendum_size:
        raise ValueError("Insufficient number of unmarked indexes to sample.")

    # 随机选择 addendum_size 个未标记索引
    selected_indices = np.random.choice(unlabeled_indices, size=addendum_size, replace=False).tolist()

    return selected_indices