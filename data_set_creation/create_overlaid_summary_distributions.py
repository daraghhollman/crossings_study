"""
Script to load the philpott boundaries list, and load and save the solar wind time series data in a fixed length window before/after the boundary, along with any other raw data such as ephemeris, local time, latitude, magnetic latitude
"""

import datetime as dt
import enum

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats

save = True

mpl.rcParams["font.size"] = 8
colours = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

# Load samples csv
ms_samples_data_set = pd.read_csv("./total_filtered_magnetosheath_sample_10_mins.csv", parse_dates=["sample_start", "sample_end", "crossing_start", "crossing_end"])
sw_samples_data_set = pd.read_csv("./total_filtered_solar_wind_sample_10_mins.csv", parse_dates=["sample_start", "sample_end", "crossing_start", "crossing_end"])

number_of_crossings = len(ms_samples_data_set)

components = ["mag_total", "mag_x", "mag_y", "mag_z"]
for component in components:
    sw_samples_data_set[component] = sw_samples_data_set[component].apply(
        lambda x: list(map(float, x.strip("[]").split(",")))
    )
    ms_samples_data_set[component] = ms_samples_data_set[component].apply(
        lambda x: list(map(float, x.strip("[]").split(",")))
    )

fig, axes = plt.subplots(1, 4, figsize=(11.7, 4), sharey=True)


names = ["SW", "MS"]
colour_indices = [0, 4]
for i, samples_data_set in enumerate([sw_samples_data_set, ms_samples_data_set]):
    for ax, component in zip(axes, components):

        # Remove extreme outliers
        data = np.array(samples_data_set[component].explode().tolist())

        outliers_removed_data = data[(np.abs(scipy.stats.zscore(data)) < 3)]

        hist_data, bin_edges = np.histogram(
            data, bins=np.arange(-300, 300, 10), density=True
        )

        ax.stairs(hist_data, bin_edges, fill=True, color=colours[i*4], alpha=0.5)

        ax.margins(0)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

        ax.axvline(np.mean(data), color=colours[colour_indices[i]], label=f"{names[i]} Mean = {np.mean(data):.2f}")
        ax.axvline(np.median(data), color=colours[colour_indices[i]], ls="dashed", label=f"      Median = {np.median(data):.2f}")

        if ax == axes[0]:
            ax.annotate(
                f"{number_of_crossings} orbits",
                xy=(0, 1),
                xycoords="axes fraction",
                size=10,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", fc="w"),
            )

        ax.legend()


x_labels = ["|B|", "B$_x$", "B$_y$", "B$_z$"]
for i in range(len(axes)):

    ax = axes[i]

    ax.set_xlabel(x_labels[i])

    if i == 0:
        ax.set_xlim(0, 300)
        ax.set_ylabel("Counts (normalised by area), binsize = 10 nT")

    else:
        ax.set_xlim(-150, 150)



"""
fig.suptitle(
    "Magnetosheath Combined Distributions\n11 ≤ LT ≤ 13;  Z$_{MSM}$ ≥ 0; 0.35 AU ≤ R$_H$ ≤ 0.4 AU; Year=2014"
)
"""

fig.tight_layout()

if save:
    plt.savefig("./total_filtered_both_distributions.pdf", dpi=500)
    plt.savefig("./total_filtered_both_distributions.png", dpi=500)
else:
    plt.show()
