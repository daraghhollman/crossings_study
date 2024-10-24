"""
Script to compare the distribution of any input data to solar wind and magnetosheath
samples for nearby local times, heliocentric distances, etc...
"""

import datetime as dt

import hermpy.mag as mag
import hermpy.plotting_tools as hermplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import spiceypy as spice

spice.furnsh("/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt")

# Input data
root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"
start_time = dt.datetime(year=2013, month=6, day=1, hour=9, minute=55)
end_time = dt.datetime(year=2013, month=6, day=1, hour=10, minute=10)

mag_buffer = dt.timedelta(minutes=30)

# Filter Parameters
local_time_range = 12  # range of local time to search either side

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


fig = plt.figure()

mag_ax = plt.subplot2grid((5, 3), (4, 0), colspan=3)

data_axes = []
sw_axes = []
ms_axes = []
for i in range(4):
    data_axes.append(plt.subplot2grid((5, 3), (0 + i, 1)))
    sw_axes.append(plt.subplot2grid((5, 3), (0 + i, 0)))
    ms_axes.append(plt.subplot2grid((5, 3), (0 + i, 2)))


# MAG Plotting
mag_ax.plot(background_data["date"], background_data["mag_total"], c="k")
mag_ax.axvline(test_data.iloc[0]["date"], ls="dashed", c=colours[0])
mag_ax.axvline(test_data.iloc[-1]["date"], ls="dashed", c=colours[0])

hermplot.Add_Tick_Ephemeris(mag_ax)

# DATA HIST PLOTTING
data_components = ["mag_total", "mag_x", "mag_y", "mag_z"]
for i, ax in enumerate(data_axes):
    # Remove extreme outliers
    data = test_data[data_components[i]]

    outliers_removed_data = data[(np.abs(scipy.stats.zscore(data)) < 3)]

    hist_data, bin_edges = np.histogram(
        data, bins=np.arange(-300, 300, 10), density=True
    )

    ax.stairs(hist_data, bin_edges, fill=True, color="grey")

    ax.margins(0)

# SOLAR WIND PLOTTING
for i, ax in enumerate(sw_axes):
    # Remove extreme outliers
    data = np.array(solar_wind_samples[components[i]].explode().tolist())

    outliers_removed_data = data[(np.abs(scipy.stats.zscore(data)) < 3)]

    hist_data, bin_edges = np.histogram(
        data, bins=np.arange(-300, 300, 10), density=True
    )

    ax.stairs(hist_data, bin_edges, fill=True, color=colours[4])

    ax.margins(0)

# MAGNETOSHEATH PLOTTING
for i, ax in enumerate(ms_axes):
    # Remove extreme outliers
    data = np.array(magnetosheath_samples[components[i]].explode().tolist())

    outliers_removed_data = data[(np.abs(scipy.stats.zscore(data)) < 3)]

    hist_data, bin_edges = np.histogram(
        data, bins=np.arange(-200, 200, 10), density=True
    )

    ax.stairs(hist_data, bin_edges, fill=True, color=colours[2])

    ax.margins(0)




plt.show()
