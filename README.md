# VizDat

## Overview 📖
**VizDat** is a lightweight and efficient visualization library designed to simplify data exploration.
It provides a quick function for visualizing data distributions, helping users identify patterns and insights effortlessly.

## Features ⚙️
- 🚀 **Fast and Simple** – Quickly generate visualizations for tabular data.
- 🌟 **User-Friendly API** – Easily integrates into any workflow.
- ⚡ **Lightweight** – Minimal dependencies for high performance.

## Installation 📦
```
pip install VizDat
```

## Usage 📊
```
import VizDat  
import pandas as pd  

df = pd.read_csv("your_data.csv")  

#Generate histograms for numerical features
VizDat.data_dist(
    data=df, bins=30, exclude=None, include=None, 
    exclude_binary=False, color="skyblue", kde_color="crimson", kde=True
)
```

### Explaination of the function and its parameters 📝
What this function does:
- Plots histograms for all numerical features in a DataFrame in a grid layout.
- Determines the number of rows and columns in the grid dynamically.
- Allows excluding or including specific features.
- Supports Kernel Density Estimate (KDE) overlay.

Parameters explanation:
- **data** (*pd.DataFrame*): The input DataFrame (required).
- **bins** (*int*): Number of bins for histograms (default: `30`).
- **exclude** (*list*): Features to exclude from visualization (default: `None`).
- **include** (*list*): Features to specifically include (default: `None`).
- **exclude_binary** (*bool*): Whether to exclude binary features (default: `False`).
- **color** (*str*): Color of the histogram bars (default: `"skyblue"`).
- **kde_color** (*str*): Color of the Kernel Density Estimate (KDE) curve (default: `"crimson"`).
- **kde** (*bool*): Whether to show the KDE curve over histograms (default: `True`).

## Contribution 🤝
We welcome contributions to improve **VizDat**! Whether it's fixing bugs, enhancing features, or improving documentation, your help is appreciated.

Please contact me via my email **mustafa.farajallah99@gmail.com** or through my linked in profile, [LinkedIn profile](https://www.linkedin.com/in/mustafa-farajallah-a274a8298/)
