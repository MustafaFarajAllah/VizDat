#necessary importations
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#----------------------------------------------



#---------------Helper functions---------------

#No. 1
def validate_data(data):
    """
    The function `validate_data` will ensure that the data is a dataframe.

    No return
    """
    if data is None:
        err = "The `data` parameter cannot be None. Please provide a valid DataFrame."
        raise ValueError(err)
    elif isinstance(data, pd.Series):
        err = "The `data` parameter must be a pandas DataFrame, not a pandas Series."
        raise ValueError(err)

    elif not isinstance(data, pd.DataFrame):
        raise TypeError("The `data` parameter must be a DataFrame.")

#No. 2
def excluded_features(data, include, exclude, exclude_binary):
    """
    Identifies numerical features with optional inclusions, exclusions, and binary feature exclusion.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing features.
        include (list, optional): List of features to include. If provided, only these features are considered.
        exclude (list, optional): List of features to exclude from the final result.
        exclude_binary (bool): If True, exclude features with exactly two unique values.

    Returns:
        list: List of filtered numerical feature names.

    Raises:
        ValueError: For invalid parameter types or non-existent features.
    """

    # A function to determine if the feature is numerical
    def is_numerical(series):
        """Check if a series is numerical."""
        return pd.api.types.is_numeric_dtype(series)
    
    # Validate the `include` parameter
    if include is not None:
        if not isinstance(include, list):
            raise ValueError("The `include` parameter must be a list of features.")
        non_existent = [f for f in include if f not in data.columns]
        if non_existent:
            raise ValueError(f"Features not found in data: {non_existent}")

    # Validate the `exclude` parameter
    if exclude is not None:
        if not isinstance(exclude, list):
            raise ValueError("The `exclude` parameter must be a list of features.")
        non_existent = [f for f in exclude if f not in data.columns]
        if non_existent:
            raise ValueError(f"Features not found in data: {non_existent}")

    # Validate the `exclude_binary` parameter
    if not isinstance(exclude_binary, bool):
        raise ValueError("The `exclude_binary` parameter must be boolean, True or False.")

    # 1. Initial feature selection
    if include:
        # Validate included features are numerical
        valid_features = [f for f in include if is_numerical(data[f])]
        non_numerical = list(set(include) - set(valid_features))
        if non_numerical:
            raise ValueError(f"Non-numerical features in include list: {non_numerical}")
        features = valid_features.copy()
    else:
        # Select all numerical features by default
        features = [f for f in data.columns if is_numerical(data[f])]

    # 2. Apply exclusions
    if exclude:
        features = [f for f in features if f not in exclude]

    # 3. Handle binary feature exclusion
    if exclude_binary:
        features = [f for f in features if len(data[f].unique()) != 2]

    return features

#---------------End of helper functions---------------



#---------------Main function(s)---------------

#No. 1
def data_dist(data, bins=30, exclude=None, include=None, exclude_binary=False, color='skyblue', kde_color='crimson', kde=True):
    """
    Plots histograms for all features in a DataFrame in a grid layout.
    The number of rows and columns is determined dynamically.

    Parameters:
        - data (*pd.DataFrame*): The input DataFrame (required).
        - bins (*int*): Number of bins for histograms (default: `30`).
        - exclude (*list*): Features to exclude from visualization (default: `None`).
        - include (*list*): Features to specifically include (default: `None`).
        - exclude_binary (*bool*): Whether to exclude binary features (default: `False`).
        - color (*str*): Color of the histogram bars (default: `"skyblue"`).
        - kde_color (*str*): Color of the Kernel Density Estimate (KDE) curve (default: `"crimson"`).
        - kde (*bool*): Whether to show the KDE curve over histograms (default: `True`).
    """
    
    validate_data(data)
    features = excluded_features(data, include, exclude, exclude_binary)
    
    # Number of features
    num_features = len(features)

    if num_features == 0:
        print("No numerical features found after filtering. Nothing to plot.")
        return
    elif num_features == 1:
        sns.histplot(x=data[features[0]], bins=bins, kde=False, color=color, edgecolor='black')

        skewness = data[features[0]].skew()
        
        if kde:
            # Calculate bin edges and bin widths
            bin_edges = np.histogram_bin_edges(data[features[0]], bins=bins)
            
            # Calculate the KDE curve
            kde_est = gaussian_kde(data[features[0]])

            # Edge case - If the data is identical (max is similar to min) this could cause an error.
            x_min = data[features[0]].min()
            x_max = data[features[0]].max()
            if x_min == x_max:
                x = np.array([x_min])  # Avoid linspace error
            else:
                x = np.linspace(x_min, x_max, 100)
    
            kde_curve = kde_est(x) * len(data[features[0]]) * np.diff(bin_edges).mean()  # Scale to match histogram counts
            
            # Overlay the KDE curve
            plt.plot(x, kde_curve, color=kde_color, lw=2, label='KDE')
        
        plt.title(f"{features[0]}\nSkewness: {skewness:.2f}")
        plt.xlabel(features[0])
        plt.ylabel("Frequency")
        plt.show()
        return
    
    # Determine grid layout
    num_cols = int(np.ceil(np.sqrt(num_features)))
    num_rows = int(np.ceil(num_features / num_cols))

    # Dynamic figsize based on number of rows and columns
    base_figsize = 5  # Base size for each subplot
    fig_width = num_cols * base_figsize
    fig_height = num_rows * base_figsize
    
    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_width, fig_height))

    axes = axes.flatten()  # Flatten 2D array to 1D
    
    # Plot histograms with KDE
    for i, column in enumerate(features):
        ax = axes[i]
        
        # Precompute histogram
        counts, edges = np.histogram(data[column], bins=bins)
        
        # Plot with Matplotlib
        ax.bar(edges[:-1], counts, width=np.diff(edges), 
               color=color, edgecolor='black', linewidth=0.5)
    
        # Add skewness
        skewness = data[column].skew()
        ax.set_title(f'{column} - Skewness: {skewness:.2f}', fontsize=14)
        ax.tick_params(axis='both', labelsize=10)
    
        if kde:
            # Calculate proper KDE scaling
            kde_est = gaussian_kde(data[column])
            x = np.linspace(data[column].min(), data[column].max(), 100) #increase the points to make smooth curve
            kde_curve = kde_est(x) * len(data[column]) * np.diff(edges).mean()
        
            ax.plot(x, kde_curve, color=kde_color, lw=2)
        
        # Clean up empty subplots
    for j in range(num_features, num_rows * num_cols):
        fig.delaxes(axes[j])
    
    # Add main title and adjust layout
    plt.suptitle('Dataset Feature Distributions with Skewness\n', fontsize=20)
    plt.tight_layout()
    plt.show()
