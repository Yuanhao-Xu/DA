import torch
from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.algorithms import select_batch

class LCMD:
    """
    Class for selecting unlabeled data indices using LCMD.
    """

    def query(self, custom_model, X_train_labeled_df, y_train_labeled_df, X_train_unlabeled_df, y_train_unlabeled_df, addendum_size):
        """
        Select indices of unlabeled data using LCMD.

        Parameters:
        - custom_model: Custom model for LCMD selection.
        - X_train_labeled_df: Labeled training data features.
        - y_train_labeled_df: Labeled training data labels.
        - X_train_unlabeled_df: Unlabeled training data features.
        - y_train_unlabeled_df: Unlabeled training data labels.
        - addendum_size: Number of indices to select.

        Returns:
        - selected_indices (list): List of selected unlabeled data indices.
        """
        # Convert DataFrame to tensors
        X_train_labeled_tensor = torch.tensor(X_train_labeled_df.values, dtype=torch.float32)
        y_train_labeled_tensor = torch.tensor(y_train_labeled_df.values, dtype=torch.float32)
        X_train_unlabeled_tensor = torch.tensor(X_train_unlabeled_df.values, dtype=torch.float32)
        y_train_unlabeled_tensor = torch.tensor(y_train_unlabeled_df.values, dtype=torch.float32)

        # Create TensorFeatureData objects
        train_data = TensorFeatureData(X_train_labeled_tensor)
        pool_data = TensorFeatureData(X_train_unlabeled_tensor)

        # Select unlabeled data using select_batch
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

        # Return the indices of selected unlabeled data
        return X_train_unlabeled_df.index[incertitude_index].tolist()
