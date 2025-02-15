# VizDat

## Overview
**VizDat** is a simple and efficient visualization library designed to make data exploration easier.  
It provides quick and intuitive functions (currently only one function) for visualizing data distributions, helping users gain insights quickly.

## Features
- **Fast and Simple** – Quickly generate visualizations for tabular data.
- **User-Friendly API** – Easily integrates into any workflow.
- **Lightweight** – Minimal dependencies for high performance.

## Installation
```
pip install vizdat
```

## Usage
```
import vizdat

#Currently, this is the only function available.
vizdat.data_dist(data=None, bins=30, exclude=None, include=None, exclude_binary=False, color='skyblue', kde_color='crimson', kde=False)
```

What this function does:
- Plots histograms for all features in a DataFrame in a grid layout.
- The number of rows and columns is determined dynamically.

Parameters explaination:
- data (pd.DataFrame): The input DataFrame
- exclude (list): Features to exclude from visualization
- include (list): Features to specifically include
- exclude_binary (bool): Whether to exclude binary features (default: False)
- kde (bool): Whether to show Kernel Density Estimate (default: False)
