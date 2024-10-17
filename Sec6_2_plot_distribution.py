import matplotlib.pyplot as plt

def plot_data_2d(X_train_full_df, y_train_full_df, labeled_indices, feature_idx=0):
    """
    Plot 2D scatter, marking labeled and unlabeled data.

    Parameters:
    - X_train_full_df: DataFrame of full training data with features.
    - y_train_full_df: DataFrame of full training data with labels.
    - labeled_indices: List of indices of labeled data.
    - feature_idx: Column index of the feature to plot, default is 0.
    """

    # Extract specified feature and label
    feature = X_train_full_df.iloc[:, feature_idx]  # Extract specified feature from feature data
    label = y_train_full_df.iloc[:, 0]  # Extract label from label data, default is column 0

    # Get features and labels of labeled data using labeled_indices
    labeled_feature = feature.loc[labeled_indices]
    labeled_label = label.loc[labeled_indices]

    # Plot the figure
    plt.figure(figsize=(10, 10))

    # Plot unlabeled data points in light gray
    plt.scatter(feature, label, color='lightgray', label='Unlabeled Data')

    # Plot labeled data points in red
    plt.scatter(labeled_feature, labeled_label, color='red', label='Labeled Data', edgecolor='black', s=100)

    # Set title and axis labels
    plt.title("Distribution of Labeled Samples in EGAL", fontsize=24)  # Title font size is 24
    plt.xlabel("Feature: Cement", fontsize=22)  # X-axis label font size is 22
    plt.ylabel("Target: Concrete Compressive Strength", fontsize=22)  # Y-axis label font size is 22

    # Set legend position to bottom right, font size is 18
    plt.legend(loc='lower right', fontsize=18)

    # Show the plot
    plt.show()
