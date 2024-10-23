"""
Script to load the philpott boundaries list, and load and save the solar wind time series data in a fixed length window before/after the boundary, along with any other raw data such as ephemeris, local time, latitude, magnetic latitude
"""

import datetime as dt

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
samples_data_set = pd.read_csv("./total_filtered_solar_wind_sample_10_mins.csv", parse_dates=["sample_start", "sample_end", "crossing_start", "crossing_end"])

number_of_crossings = len(samples_data_set)

components = ["mag_total", "mag_x", "mag_y", "mag_z"]
for component in components:
    samples_data_set[component] = samples_data_set[component].apply(
        lambda x: list(map(float, x.strip("[]").split(",")))
    )

fig, axes = plt.subplots(1, 4, figsize=(11.7, 4), sharey=True)


# Fix loading issues to do with an element being a series itself
for ax, component in zip(axes, components):

    # Remove extreme outliers
    #data = np.array(samples_data_set[component].tolist())
    data = np.array(samples_data_set[component].explode().tolist())

    outliers_removed_data = data[(np.abs(scipy.stats.zscore(data)) < 3)]

    hist_data, bin_edges = np.histogram(
        outliers_removed_data, bins=np.arange(-50, 100, 5), density=True
    )

    ax.stairs(hist_data, bin_edges, fill=True, color="black")

    ax.margins(0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    ax.axvline(np.mean(data), color=colours[2], label=f"Mean = {np.mean(data):.2f}")
    ax.axvline(np.median(data), color=colours[3], label=f"Median = {np.median(data):.2f}")

    if ax == axes[0]:
        ax.annotate(
            f"{number_of_crossings} orbits\nN={len(data)}",
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
        ax.set_xlim(0, 100)
        ax.set_ylabel("Counts (normalised by area), binsize = 5 nT")


    else:
        ax.set_xlim(-50, 50)


"""
fig.suptitle(
    "Solar Wind Combined Distributions\n10 minute window beside bowshock\n11 ≤ LT ≤ 13;  Z$_{MSM}$ ≥ 0; 0.35 AU ≤ R$_H$ ≤ 0.4 AU; Year=2014"
)
"""
fig.suptitle(
    "Solar Wind Combined Distributions"
)

fig.tight_layout()

if save:
    plt.savefig("./total_filtered_sw_distributions.pdf", dpi=500)
    plt.savefig("./total_filtered_sw_distributions.png", dpi=500)
else:
    plt.show()
