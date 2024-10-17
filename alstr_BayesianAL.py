import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import torch.nn as nn
from pyro.nn import PyroModule, PyroSample
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Define Bayesian Neural Network
class BNN(PyroModule):
    def __init__(self, in_dim=8, out_dim=1, hid_dims=[64, 32], prior_scale=5.):
        super().__init__()

        self.activation = nn.ReLU()  # Use ReLU activation function
        assert in_dim > 0 and out_dim > 0 and len(hid_dims) > 0  # Ensure valid dimensions

        # Define layer sizes: input + hidden + output dimensions
        self.layer_sizes = [in_dim] + hid_dims + [out_dim]

        # Create list of layers
        layer_list = [
            PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx])
            for idx in range(1, len(self.layer_sizes))
        ]

        # Convert layer list to ModuleList in PyroModule
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        # Define priors for weights and biases of each layer
        for idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(
                dist.Normal(0., prior_scale * np.sqrt(2 / self.layer_sizes[idx]))
                .expand([self.layer_sizes[idx + 1], self.layer_sizes[idx]])
                .to_event(2)
            )
            layer.bias = PyroSample(
                dist.Normal(0., prior_scale)
                .expand([self.layer_sizes[idx + 1]])
                .to_event(1)
            )

    def forward(self, x, y=None):
        x = x.reshape(-1, self.layer_sizes[0])  # Ensure correct input dimensions
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))  # Pass through hidden layers
        mu = self.layers[-1](x).squeeze(-1)  # Ensure mu is a 1D vector [batch_size]
        sigma = pyro.sample("sigma", dist.Gamma(2.0, 1.0))  # Sample scalar sigma

        # Broadcast sigma to match the shape of mu
        sigma = sigma.expand_as(mu)

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)

        return mu

# Define Active Learning class
class BayesianAL:
    def __init__(self,
                 hid_dims=[64, 32],  # Default hidden dimensions
                 prior_scale=5.0):  # Default prior scale

        # Input and output dimensions will be set in query method
        self.hid_dims = hid_dims
        self.prior_scale = prior_scale

        # Initialize BNN components but not the data
        self.guide = None
        self.svi = None

    def initialize_model(self, input_dim, output_dim):
        # Initialize BNN and Pyro components
        self.bnn = BNN(in_dim=input_dim, out_dim=output_dim, hid_dims=self.hid_dims, prior_scale=self.prior_scale)
        self.guide = pyro.infer.autoguide.AutoDiagonalNormal(self.bnn)
        self.optimizer = pyro.optim.Adam({"lr": 0.001})
        self.svi = pyro.infer.SVI(self.bnn, self.guide, self.optimizer, loss=pyro.infer.Trace_ELBO())

    def train(self, X_train_labeled_tensor, y_train_labeled_tensor, num_iterations=1500):
        for j in range(num_iterations):
            loss = self.svi.step(X_train_labeled_tensor, y_train_labeled_tensor)

    def predict_with_uncertainty(self, X, num_samples=50):
        sampled_models = [self.guide() for _ in range(num_samples)]
        yhats = [self.bnn(X).detach().numpy() for model in sampled_models]
        mean = np.mean(yhats, axis=0)
        uncertainty = np.std(yhats, axis=0)
        return mean, uncertainty

    def query(self, X_train_unlabeled_df, X_train_labeled_df, y_train_labeled_df, addendum_size):
        # Initialize model based on input data dimensions
        input_dim = X_train_labeled_df.shape[1]
        output_dim = y_train_labeled_df.shape[1]
        self.initialize_model(input_dim, output_dim)

        # Convert labeled and unlabeled data to tensors
        X_train_labeled_tensor = torch.tensor(X_train_labeled_df.values, dtype=torch.float32)
        y_train_labeled_tensor = torch.tensor(y_train_labeled_df.values, dtype=torch.float32)
        X_train_unlabeled_tensor = torch.tensor(X_train_unlabeled_df.values, dtype=torch.float32)

        # Train model
        self.train(X_train_labeled_tensor, y_train_labeled_tensor)

        # Compute uncertainty and select most uncertain samples
        _, uncertainties = self.predict_with_uncertainty(X_train_unlabeled_tensor)
        uncertainty_indices = np.argsort(uncertainties.flatten())[::-1][:addendum_size]
        selected_indices = X_train_unlabeled_df.index[uncertainty_indices].tolist()  # Convert to absolute indices

        return selected_indices  # Return selected indices
