import numpy as np
import pandas as pd

class DataGenerator:
    def __init__(self, n_samples=1000, n_features=1, n_lin=0, noise=0.05, seed=50, phase=True):
        """
        Initialize DataGenerator with sample size, feature count, linear feature count, noise factor, and seed.
        The 'phase' parameter controls whether to add phase shift, default is False.
        The 'n_lin' parameter controls the number of linear features.
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_lin = n_lin
        self.noise = noise
        self.seed = seed
        self.phase = phase

        # Ensure linear feature count does not exceed total features
        assert 0 <= self.n_lin <= self.n_features, "Linear feature count cannot exceed total feature count."

    def generate_data(self, noise_level=0):
        """
        Generate dataset with some features being linear and others nonlinear (sine function).
        The 'noise_level' parameter allows generating different noise levels using different random seeds.
        """
        # Set main seed and generate feature data
        np.random.seed(self.seed)

        # Generate random feature matrix X, range [0, 2π], with some linear features
        X = np.random.rand(self.n_samples, self.n_features) * 2 * np.pi

        # Generate linear features (first n_lin are linear)
        X_linear = X[:, :self.n_lin]

        # Nonlinear features use sine function
        X_nonlinear = X[:, self.n_lin:]

        if self.phase:
            # Add phase shift to nonlinear features
            phase_shifts = np.random.rand(self.n_features - self.n_lin) * 2 * np.pi
            y_nonlinear = np.sum(np.sin(X_nonlinear + phase_shifts), axis=1)
        else:
            # No phase shift
            y_nonlinear = np.sum(np.sin(X_nonlinear), axis=1)

        # Linearly combine linear features
        y_linear = np.sum(X_linear, axis=1)

        # Final target y is a combination of linear and nonlinear parts
        y = y_linear + y_nonlinear

        # Generate noise with a seed based on the initial seed and noise_level
        noise_seed = self.seed + noise_level  # Use different seed for noise
        np.random.seed(noise_seed)
        y_std = np.std(y)
        noise = np.random.normal(0, self.noise * y_std, size=y.shape)
        y += noise

        # Convert data to Pandas DataFrame
        self.data = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(self.n_features)])
        self.data['target'] = y

        # Round values to 4 decimals
        self.data = self.data.round(4)

    def save_to_csv(self, file_name):
        """
        Save generated data to a CSV file.
        """
        if hasattr(self, 'data'):
            self.data.to_csv(file_name, index=False)
            print(f"Data saved to {file_name}")
        else:
            print("No data generated. Please run generate_data() first.")

# Example usage for generating datasets:
# gen = DataGenerator(n_samples=1100, n_features=7, n_lin=0, noise=0.10, seed=50)
# gen.generate_data(noise_level=0)
# gen.save_to_csv('data_1100s_7f10n.csv')

# gen = DataGenerator(n_samples=1100, n_features=7, n_lin=0, noise=0.15, seed=50)
# gen.generate_data(noise_level=1)
# gen.save_to_csv('data_1100s_7f15n.csv')

# gen = DataGenerator(n_samples=1100, n_features=7, n_lin=0, noise=0.20, seed=50)
# gen.generate_data(noise_level=2)
# gen.save_to_csv('data_1100s_7f20n.csv')

# gen = DataGenerator(n_samples=1100, n_features=9, n_lin=0, noise=0.05, seed=50)
# gen.generate_data(noise_level=0)
# gen.save_to_csv('data_1100s_9f5n.csv')

# gen = DataGenerator(n_samples=1100, n_features=11, n_lin=0, noise=0.05, seed=50)
# gen.generate_data(noise_level=0)
# gen.save_to_csv('data_1100s_11f5n.csv')
