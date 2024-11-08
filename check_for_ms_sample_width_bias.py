"""
Script to plot the distributions of grazing angle and IMF |B| for bow shock crossings for:
    a) All crossings
    b) Crossings where we have a 10 minute window in both the solar wind and the magnetosheath

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load solar wind and magnetosheath samples
solar_wind_samples = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/solar_wind_sample_10_mins.csv"
)
solar_wind_features = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/solar_wind_features.csv"
)

solar_wind_features["mean"] = solar_wind_features["mean"].apply(
    lambda s: list(map(float, s.strip("[]").split()))
)
solar_wind_features["Mean IMF |B|"] = np.array(solar_wind_features["mean"].tolist())[
    :, 0
]

magnetosheath_samples = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/magnetosheath_sample_10_mins.csv"
)
magnetosheath_features = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/magnetosheath_features.csv"
)

magnetosheath_samples["|B|"] = magnetosheath_samples["|B|"].apply(
    lambda x: list(map(float, x.strip("[]").split(",")))
)

indices_to_keep = []
# loop through the crossings
for i in range(len(solar_wind_samples)):

    magnetosheath_length = len(magnetosheath_samples.iloc[i]["|B|"])

    if magnetosheath_length >= 600:
        indices_to_keep.append(i)

# First plot the distribution of grazing angle, and IMF |B| for all crossings
fig, axes = plt.subplots(1, 2, sharey=True)

# Remove outliers
grazing_angle_bins = np.arange(0, np.max(solar_wind_features["grazing_angle"]) + 1, 1)
mean_imf_bins = np.arange(0, np.max(solar_wind_features["Mean IMF |B|"]) + 2, 2)

axes[0].hist(
    solar_wind_features["grazing_angle"],
    bins=grazing_angle_bins,
    density=True,
    color="indianred",
    alpha=0.5,
)
axes[1].hist(
    solar_wind_features["Mean IMF |B|"],
    bins=mean_imf_bins,
    density=True,
    color="indianred",
    alpha=0.5,
    label="All",
)

# Set limits based on z score
z = 3
axes[0].set_xlim(
    np.mean(solar_wind_features["grazing_angle"])
    - z * np.std(solar_wind_features["grazing_angle"]),
    np.mean(solar_wind_features["grazing_angle"])
    + z * np.std(solar_wind_features["grazing_angle"]),
)
axes[1].set_xlim(
    0,
    np.mean(solar_wind_features["Mean IMF |B|"])
    + z * np.std(solar_wind_features["Mean IMF |B|"]),
)

axes[0].set_xlabel("Grazing Angle")
axes[0].set_ylabel("fraction of bow shock crossings per bin")

axes[1].set_xlabel("Mean IMF |B|")

axes[0].annotate(
    f"N (all) = {len(solar_wind_features)}\nN (filtered) = {len(solar_wind_features.iloc[indices_to_keep])}",
    xy=(1, 1),
    xycoords="axes fraction",
    size=10,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round", fc="w"),
)


# Then overplot the subset distributions
axes[0].hist(
    solar_wind_features.iloc[indices_to_keep]["grazing_angle"],
    bins=grazing_angle_bins,
    density=True,
    color="cornflowerblue",
    alpha=0.5,
)
axes[1].hist(
    solar_wind_features.iloc[indices_to_keep]["Mean IMF |B|"],
    bins=mean_imf_bins,
    density=True,
    color="cornflowerblue",
    alpha=0.5,
    label="Filtered",
)

axes[1].legend()

plt.show()
