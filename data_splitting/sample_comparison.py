"""
Script to compare the distribution of any input data to solar wind and magnetosheath
samples for nearby local times, heliocentric distances, etc...
"""

import datetime as dt

import hermpy.boundary_crossings as boundaries
import hermpy.mag as mag
import hermpy.plotting_tools as hermplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import spiceypy as spice
from hotelling.stats import hotelling_t2
from tqdm import tqdm

spice.furnsh("/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt")
crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/Work/mercury/philpott_2020_reformatted.csv"
)

method = "None"  # T2, KS, PBP (point by point)

# Input data
root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"
start_time = dt.datetime(year=2012, month=7, day=2, hour=17, minute=36)
end_time = dt.datetime(year=2012, month=7, day=2, hour=17, minute=37)

mag_buffer = dt.timedelta(minutes=10)

# Filter Parameters
local_time_range = 1  # range of local time to search either side

# Plotting settings
save = True
mpl.rcParams["font.size"] = 8
colours = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

# Load samples csv
magnetosheath_samples = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/crossings_study/data_set_creation/magnetosheath_sample_10_mins.csv",
    parse_dates=["sample_start", "sample_end", "crossing_start", "crossing_end"],
)
solar_wind_samples = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/crossings_study/data_set_creation/solar_wind_sample_10_mins.csv",
    parse_dates=["sample_start", "sample_end", "crossing_start", "crossing_end"],
)

components = ["|B|", "B_x", "B_y", "B_z"]
for component in components:
    solar_wind_samples[component] = solar_wind_samples[component].apply(
        lambda x: list(map(float, x.strip("[]").split(",")))
    )
    magnetosheath_samples[component] = magnetosheath_samples[component].apply(
        lambda x: list(map(float, x.strip("[]").split(",")))
    )

# Load test data
test_data = mag.Load_Between_Dates(root_dir, start_time, end_time, strip=True)

background_data = mag.Load_Between_Dates(
    root_dir, start_time - mag_buffer, end_time + mag_buffer, strip=True
)


# Get middle parameters of data
middle_data = test_data.iloc[len(test_data) // 2]
longitude = np.arctan2(middle_data["eph_y"], middle_data["eph_x"]) * 180 / np.pi

if longitude < 0:
    longitude += 360

local_time = ((longitude + 180) * 24 / 360) % 24


# Limit samples by paramters of data
filtered_solar_wind_samples = solar_wind_samples.loc[
    (solar_wind_samples["LT"] < local_time + local_time_range)
    & (solar_wind_samples["LT"] > local_time - local_time_range)
]
filtered_magnetosheath_samples = magnetosheath_samples.loc[
    (magnetosheath_samples["LT"] < local_time + local_time_range)
    & (magnetosheath_samples["LT"] > local_time - local_time_range)
]


fig = plt.figure(figsize=(11.7, 8.3))

mag_ax = plt.subplot2grid((4, 4), (3, 0), colspan=4)

data_axes = []
sw_axes = []
ms_axes = []
for i in range(4):
    sw_axes.append(plt.subplot2grid((4, 4), (0, i)))
    data_axes.append(plt.subplot2grid((4, 4), (1, i)))
    ms_axes.append(plt.subplot2grid((4, 4), (2, i)))


# MAG Plotting
mag_ax.plot(background_data["date"], background_data["mag_total"], c="k")

mag_ax.plot(
    background_data["date"],
    background_data["mag_x"],
    color=colours[2],
    lw=0.7,
    alpha=0.8,
    label="Bx",
)
mag_ax.plot(
    background_data["date"],
    background_data["mag_y"],
    color=colours[0],
    lw=0.7,
    alpha=0.8,
    label="By",
)
mag_ax.plot(
    background_data["date"],
    background_data["mag_z"],
    color=colours[-1],
    lw=0.7,
    alpha=0.8,
    label="Bz",
)
boundaries.Plot_Crossing_Intervals(
    mag_ax,
    start_time - mag_buffer,
    end_time + mag_buffer,
    crossings,
    color=colours[3],
    lw=2,
)
mag_ax.margins(0)

mag_leg = mag_ax.legend(
    bbox_to_anchor=(1, 1), loc="upper right", ncol=4, borderaxespad=0.5
)
vline = mag_ax.axvline(test_data.iloc[0]["date"], ls="dashed", c="grey")
mag_ax.axvline(test_data.iloc[-1]["date"], ls="dashed", c="grey")
mag_ax.axvspan(
    test_data.iloc[0]["date"],
    test_data.iloc[-1]["date"],
    color="grey",
    alpha=0.3,
)

hermplot.Add_Tick_Ephemeris(mag_ax)

data_components = ["mag_total", "mag_x", "mag_y", "mag_z"]

if method == "T2":
    data_features = []
    sw_features = []
    ms_features = []

if method == "PBP":
    probabilities = {
        "mag_total": [],
        "mag_x": [],
        "mag_y": [],
        "mag_z": [],
    }

for i in range(len(data_axes)):
    # DATA HIST PLOTTING
    data = test_data[data_components[i]]

    # Remove extreme outliers
    outliers_removed_data = data[(np.abs(scipy.stats.zscore(data)) < 3)]

    hist_data, bin_edges = np.histogram(
        data, bins=np.arange(-200, 200 + 10, 10), density=True
    )

    data_axes[i].stairs(hist_data, bin_edges, fill=True, color="grey")
    data_axes[i].margins(0)

    # SOLAR WIND PLOTTING
    # Remove extreme outliers
    sw_data = np.array(solar_wind_samples[components[i]].explode().tolist())

    sw_outliers_removed_data = sw_data[(np.abs(scipy.stats.zscore(sw_data)) < 3)]

    sw_hist_data, bin_edges = np.histogram(
        sw_data, bins=np.arange(-200, 200 + 10, 10), density=True
    )

    sw_axes[i].stairs(sw_hist_data, bin_edges, fill=True, color=colours[4])
    sw_axes[i].margins(0)

    # MAGNETOSHEATH PLOTTING
    # Remove extreme outliers
    ms_data = np.array(magnetosheath_samples[components[i]].explode().tolist())

    ms_outliers_removed_data = ms_data[(np.abs(scipy.stats.zscore(ms_data)) < 3)]

    ms_hist_data, bin_edges = np.histogram(
        ms_data, bins=np.arange(-200, 200 + 10, 10), density=True
    )

    ms_axes[i].stairs(ms_hist_data, bin_edges, fill=True, color=colours[2])
    ms_axes[i].margins(0)

    if method == "T2":
        data_features.append(data)
        sw_features.append(sw_data)
        ms_features.append(ms_data)

    if method == "KS":
        data_sw_test = scipy.stats.kstest(data, sw_data)
        data_ms_test = scipy.stats.kstest(data, ms_data)
        print(data_sw_test)
        print(data_ms_test)

    if method == "PBP":
        # For each variable (the current loop), we loop through the entire time series
        # and determine the probability that that point belongs to MS or SW.
        # (Perhaps in the form of a probability vector, -1 is MS, +1 is SW)
        # We add that probability vector to a list correspondng to that variable.
        # We then can combine the lists in some way (possibly weighted by variable)
        # to yield one list to colour the time series.

        # Get the PDFs of the SW and MS distributions using KDE (Kernel Density Estimation)
        sw_kde = scipy.stats.gaussian_kde(sw_data)
        ms_kde = scipy.stats.gaussian_kde(ms_data)

        for point in tqdm(data):

            # Determine likelihood
            sw_likelihood = sw_kde(point)
            ms_likelihood = ms_kde(point)

            probabilities[data_components[i]].append(ms_likelihood - sw_likelihood)


if method == "T2":

    data_sw_test = hotelling_t2(np.array(sw_features).T, np.array(data_features).T)
    data_ms_test = hotelling_t2(np.array(ms_features).T, np.array(data_features).T)

    print(data_sw_test)
    print(data_ms_test)


if method == "PBP":
    plt.show()

    for component in data_components:
        plt.plot(np.arange(0, len(probabilities[component])), probabilities[component])

    plt.show()


axes = np.reshape(fig.axes[1:], (4, 3))
for i in range(len(axes[0])):

    max_y = 0

    for j in range(len(axes[:, 0])):

        if max_y < axes[j][i].get_ylim()[1]:
            max_y = axes[j][i].get_ylim()[1]

        if i == 0:
            axes[j][i].text(0.5, 1.2, components[j], transform=axes[j][i].transAxes, size=12, ha="center", va="center")

        if i != 2:
            axes[j][i].set_xticklabels([])

        if j == 0:
            if i == 1:
                axes[j][i].set_ylabel("Counts (normalised by area), 10 nT bins")

            region_labels = ["Region Average\nSolar Wind", "Data Window", "Region Average\nMagnetosheath"]
            axes[j][i].text(-0.5, 0.5, region_labels[i], transform=axes[j][i].transAxes, size=12, ha="center", va="center", rotation=90)

        else:
            axes[j][i].set_yticklabels([])

    for j in range(len(axes[:, 0])):
        axes[j][i].set_ylim(0, max_y)


if save:
    plt.savefig(
        f"/home/daraghhollman/Main/Work/mercury/Figures/Splitting/{start_time.strftime('%Y%m%d_%H%M')}.png",
        dpi=300,
    )
else:
    plt.show()
