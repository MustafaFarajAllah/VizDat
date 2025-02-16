#necessary importations
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def validate_data(data):
    """
    The function `validate_data` will ensure that the data is a dataframe.

    No return...
    """
    if data is None:
        err = "The `data` parameter cannot be None. Please provide a valid DataFrame."
        raise ValueError(err)
    elif isinstance(data, pd.Series):
        err = "The `data` parameter must be a pandas DataFrame, not a pandas Series."
        raise ValueError(err)

    elif not isinstance(data, pd.DataFrame):
        raise TypeError("The `data` parameter must be a DataFrame.")

#---------------------

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
    
    def is_numerical(series):
        """Check if a series is numerical."""
        return pd.api.types.is_numeric_dtype(series)
    
    # Validate input parameters
    if include is not None:
        if not isinstance(include, list):
            raise ValueError("The `include` parameter must be a list of features.")
        non_existent = [f for f in include if f not in data.columns]
        if non_existent:
            raise ValueError(f"Features not found in data: {non_existent}")
    
    if exclude is not None:
        if not isinstance(exclude, list):
            raise ValueError("The `exclude` parameter must be a list of features.")
        non_existent = [f for f in exclude if f not in data.columns]
        if non_existent:
            raise ValueError(f"Features not found in data: {non_existent}")

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

def data_dist(data=None, bins=30, exclude=None, include=None, exclude_binary=False, color='skyblue', kde_color='crimson', kde=False):
    """
    Plots histograms for all features in a DataFrame in a grid layout.
    The number of rows and columns is determined dynamically.

    Parameters:
        data (pd.DataFrame): The input DataFrame
        exclude (list): Features to exclude from visualization
        include (list): Features to specifically include
        exclude_binary (bool): Whether to exclude binary features (default: False)
        kde (bool): Whether to show Kernel Density Estimate (default: False)
    """
    
    validate_data(data)
    features = excluded_features(data, include, exclude, exclude_binary)

    if not features:
        raise ValueError("No numerical features found after filtering. Adjust `include` and `exclude` parameters.")
    
    # Number of features
    num_features = len(features)
    
    # Determine grid layout
    num_cols = int(np.ceil(np.sqrt(num_features)))
    num_rows = int(np.ceil(num_features / num_cols))

    # Dynamic figsize based on number of rows and columns
    base_figsize = 3  # Base size for each subplot
    fig_width = num_cols * base_figsize
    fig_height = num_rows * base_figsize
    
    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_width, fig_height))

    if num_features == 1:
        axes = [axes]  # Convert single axis to list
    else:
        axes = axes.flatten()
    
    # Plot histograms with KDE
    for i, column in enumerate(features):
        ax = axes[i]
        
        # Precompute histogram
        counts, edges = np.histogram(data[column], bins=bins)
        bin_width = edges[1] - edges[0]  # Calculate bin width
        
        # Plot with Matplotlib
        ax.bar(edges[:-1], counts, width=np.diff(edges), 
               color=color, edgecolor='black', linewidth=0.5)
    
        # Add skewness
        skewness = data[column].skew()
        ax.set_title(f'{column}\nSkewness: {skewness:.2f}', fontsize=9)
        ax.tick_params(axis='both', labelsize=6)
    
        if kde:
            # Calculate proper KDE scaling
            kde_est = gaussian_kde(data[column])
            x = np.linspace(edges[0], edges[-1], 100)  # Increased points for smoother curve
            kde_curve = kde_est(x) * (bin_width * len(data[column]))  # Correct scaling
        
            ax.plot(x, kde_curve, color=kde_color, lw=2)
        
        # Clean up empty subplots
    for j in range(num_features, num_rows * num_cols):
        fig.delaxes(axes[j])
    
    # Add main title and adjust layout
    plt.suptitle('Dataset Feature Distributions with Skewness', fontsize=14)
    plt.tight_layout()
    plt.show()