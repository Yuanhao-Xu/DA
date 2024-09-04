import torch
from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.algorithms import select_batch

class LCMD:
    """
    用于使用 LCMD 方法选择未标记数据索引的类。
    """

    def query(self, custom_model, X_train_labeled_df, y_train_labeled_df, X_train_unlabeled_df, y_train_unlabeled_df, addendum_size):
        """
        使用 LCMD 方法选择未标记数据的索引。

        参数:
        - custom_model: 用于 LCMD 选择的自定义模型。
        - X_train_labeled_df: 已标记训练数据的特征 DataFrame。
        - y_train_labeled_df: 已标记训练数据的标签 DataFrame。
        - X_train_unlabeled_df: 未标记训练数据的特征 DataFrame。
        - y_train_unlabeled_df: 未标记训练数据的标签 DataFrame。
        - addendum_size: 本次选择的索引数量。

        返回:
        - selected_indices (list): 本次选择的未标记数据的索引列表。
        """
        # 将 DataFrame 转换为 tensor
        X_train_labeled_tensor = torch.tensor(X_train_labeled_df.values, dtype=torch.float32)
        y_train_labeled_tensor = torch.tensor(y_train_labeled_df.values, dtype=torch.float32)
        X_train_unlabeled_tensor = torch.tensor(X_train_unlabeled_df.values, dtype=torch.float32)
        y_train_unlabeled_tensor = torch.tensor(y_train_unlabeled_df.values, dtype=torch.float32)

        # 构造 TensorFeatureData 对象
        train_data = TensorFeatureData(X_train_labeled_tensor)
        pool_data = TensorFeatureData(X_train_unlabeled_tensor)

        # 使用 select_batch 选择未标记数据
        incertitude_index, _ = select_batch(
            batch_size=addendum_size,
            models=[custom_model],
            data={'train': train_data, 'pool': pool_data},
            y_train=y_train_labeled_tensor,
            selection_method='lcmd',
            sel_with_train=True,
            base_kernel='grad',
            kernel_transforms=[('rp', [512])]
        )

        # 返回选中的未标记数据集的索引
        return X_train_unlabeled_df.index[incertitude_index].tolist()
