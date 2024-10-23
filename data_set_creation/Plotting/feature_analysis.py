"""
Script to load data from the features dataset and make basic plots
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats

save = False

mpl.rcParams["font.size"] = 12

# Load samples csv 
features_data = pd.read_csv("./noon_north_filtered_solar_wind_features.csv")

# Fix loading issues to do with an element being a series itself
features = ["mean", "median", "std", "skew", "kurtosis"]
variables = ["|B|", "B$_x$", "B$_y$", "B$_z$"]

for feature in features:
    features_data[feature] = features_data[feature].apply(lambda x: list(map(float, x.strip("[]").strip().split())))


# Plot distributions for each variable for each component
fig, axes = plt.subplots(len(features_data[features[0]].iloc[0]), len(features), sharex="col", sharey="col", figsize=(11.7, 6))

for i in range(len(features)):
    for j in range(len(features_data[features[i]].iloc[0])):

        # Remove extreme outliers
        data = np.array(features_data[features[i]].tolist())[:,j]
        outliers_removed_data = data[ (np.abs(scipy.stats.zscore(data)) < 3) ]

        hist_data, bin_edges = np.histogram(outliers_removed_data, bins=14, density=True)

        ax = axes[j][i]
        ax.stairs(hist_data, bin_edges, fill=True, color="black")

        ax.set_yticks([])
        ax.margins(0)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

        # Applies to the last row
        if j == len(features_data[features[i]].iloc[0]) - 1:
            ax.set_xlabel(features[i])

        # Applies to the first column
        if i == 0:
            ax.text(-0.3, 0.5, variables[j], ha="center", va="center", transform=ax.transAxes, rotation="horizontal", fontsize="x-large")


plt.suptitle("Distributions of solar wind parameters for 10 minute windows")

if save:
    plt.savefig("./noon_north_filtered_features_distributions.pdf", dpi=300)
    plt.savefig("./noon_north_filtered_features_distributions.png", dpi=300)
else:
    plt.show()
