import numpy as np

class RS:
    def query(self, unlabeled_indices, addendum_size):
        if len(unlabeled_indices) < addendum_size:
            raise ValueError("Insufficient number of unmarked indexes to sample.")

        # Randomly select 'addendum_size' indices
        selected_indices = np.random.choice(unlabeled_indices, size=addendum_size, replace=False).tolist()

        return selected_indices
