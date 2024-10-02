"""
Script to create histograms depicting the distribution of crossing intervals in heliocentric distance and local time.

For each crossing interval, we determine the midpoint and calculate Mercury's distance to the sun and the local time of the spacecraft at that time.
"""

import datetime as dt

import hermpy.boundary_crossings as boundaries
import hermpy.trajectory as trajectory
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from tqdm import tqdm
import scipy.stats

mpl.rcParams["font.size"] = 14

# Import nessessary files
spice.furnsh("/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt")


# Import the crossing interval list
crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/mercury/philpott_2020_reformatted.csv"
)


# Determine distance and local time for each crossing midpoint
heliocentric_distances = np.empty(len(crossings), dtype=dt.datetime)
local_times = np.empty(len(crossings), dtype=dt.datetime)

pos_a = []
pos_b = []

for i in tqdm(range(len(crossings)), total=len(crossings)):

    midpoint = crossings.iloc[i]["start"] + (
        crossings.iloc[i]["end"] - crossings.iloc[i]["start"]
    )

    # Heliocentric distance
    heliocentric_distances[i] = trajectory.Get_Heliocentric_Distance(midpoint)

    # Convert from km to units of Mercury's semi-major axis
    heliocentric_distances[i] /= 57.91e6

    # Local time calculation
    position_start = trajectory.Get_Position("MESSENGER", crossings.iloc[i]["start"])
    position_end = trajectory.Get_Position("MESSENGER", crossings.iloc[i]["end"])

    position = position_start

    if position is None:
        continue

    longitude = np.arctan2(position[1], position[0]) * 180 / np.pi

    if longitude < 0:
        longitude += 360

    local_times[i] = ((longitude + 180) * 24 / 360) % 24


# Remove unassigned elements
heliocentric_distances = heliocentric_distances[heliocentric_distances != None]
local_times = local_times[local_times != None]


# We also want to get the heliocentric distances for the whole mission
# This will help us to isolate bias from the timing of the mission
mission_start = dt.datetime(year=2011, month=3, day=23)
mission_end = dt.datetime(year=2015, month=4, day=30)

# Sample every hour
# Initially I was sampling every day, but got some weird resonance effects in the data
mission_hours = [
    mission_start + i * dt.timedelta(hours=1)
    for i in range((mission_end - mission_start).days * 24 + 1)
]
mission_heliocentric_distances = np.array(
    [trajectory.Get_Heliocentric_Distance(date) / 57.91e6 for date in mission_hours]
)

# The same for local time! Just in case
mission_local_times = []
for i in range(len(mission_hours)):

    position = trajectory.Get_Position("MESSENGER", mission_start + i * dt.timedelta(hours=1))

    if position is None:
        continue

    longitude = np.arctan2(position[1], position[0]) * 180 / np.pi

    if longitude < 0:
        longitude += 360

    mission_local_times.append(((longitude + 180) * 24 / 360) % 24)

# Make histograms
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

ax1, ax2, ax3 = axes

heliocentric_bins = np.arange(0.79, 1.21 + 0.01, 0.01)
local_time_bins = np.arange(0, 24 + 1, 1)

cmap = mpl.colormaps["viridis"]

# Total mission heliocentric distribution
mission_hist_data_heliocentric, heliocentric_bin_edges, _ = ax1.hist(
    mission_heliocentric_distances,
    bins=heliocentric_bins,
    label=f"{len(mission_hours)} hours",
    color=cmap(0.25),
    density=True,
)

# Raw interval samples for heliocentric distance
intervals_hist_data, _, _ = ax2.hist(
    heliocentric_distances,
    bins=heliocentric_bins,
    label=f"{len(crossings)} crossing intervals",
    color=cmap(0.5),
    density=True,
)


# Perform T-Test
heliocentric_t_test = scipy.stats.ttest_ind(mission_hist_data_heliocentric, intervals_hist_data)

# Raw interval samples divided by total mission
heliocentric_bin_centres = (
    heliocentric_bin_edges[:-1] + heliocentric_bin_edges[1:]
) / 2

ratio_heliocentric = np.divide(intervals_hist_data, mission_hist_data_heliocentric)
ax3.plot(
    heliocentric_bin_centres,
    ratio_heliocentric,
    color=cmap(0.75),
)

ax3.axhline(
    np.nanmean(ratio_heliocentric), color="grey", label=r"$\mu$" + f" = {np.nanmean(ratio_heliocentric):.2f}"
)
ax3.axhline(
    np.nanmean(ratio_heliocentric) + np.nanstd(ratio_heliocentric),
    color="grey",
    ls="dotted",
    label=r"$\sigma$" + f" = {np.nanstd(ratio_heliocentric):.3f}",
)
ax3.axhline(np.nanmean(ratio_heliocentric) - np.nanstd(ratio_heliocentric), color="grey", ls="dotted")


ax1.set_xlabel("Heliocentric Distance [ semi-major axes ]\n(Binsize: 0.01)")
ax1.set_ylabel("Probability Density")
ax1.set_title("Full Mission Distribution\n(Sampled Hourly)")

ax2.set_xlabel("Heliocentric Distance [ semi-major axes ]\n(Binsize: 0.01)")
ax2.set_ylabel("Probability Density")
ax2.set_title("Boundary Interval Distribution\n(Start Time)")

ax3.set_ylim(0.8, 1.2)
ax3.set_xlabel("Heliocentric Distance [ semi-major axes ]\n(Binsize: 0.01)")
ax3.set_ylabel("Ratio (b / a)")
ax3.set_title(f"Normalised Boundary Interval Distribution\n(Panels b / a)\nT-Test: p = {heliocentric_t_test.pvalue}")


axis_labels = ["a", "b", "c"]
for i, ax in enumerate(axes):
    ax.text(0.05, 0.95, axis_labels[i] + ")", transform=ax.transAxes)
    ax.margins(0)
    ax.legend()

plt.savefig("/home/daraghhollman/Main/mercury/Figures/philpott_heliocentric_distributions_extended.png", dpi=200)


# Repeat for local time
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

ax1, ax2, ax3 = axes

# Total mission local time distribution
mission_hist_data_local_time, local_time_bin_edges, _ = ax1.hist(
    mission_local_times,
    bins=local_time_bins,
    label=f"{len(mission_hours)} hours",
    color=cmap(0.25),
    density=True,
)


# Raw interval samples for local time
intervals_hist_data_local_time, _, _ = ax2.hist(
    local_times,
    bins=local_time_bins,
    label=f"{len(crossings)} crossing intervals",
    color=cmap(0.5),
    density=True,
)

# Perform T-Test
local_time_t_test = scipy.stats.ttest_ind(mission_hist_data_heliocentric, intervals_hist_data_local_time)

# Raw interval samples divided by total mission
local_time_bin_centres = (
    local_time_bin_edges[:-1] + local_time_bin_edges[1:]
) / 2

ratio_local_time = np.divide(intervals_hist_data_local_time, mission_hist_data_local_time)
ax3.plot(
    local_time_bin_centres,
    ratio_local_time,
    color=cmap(0.75),
)

ax3.axhline(
    np.nanmean(ratio_local_time), color="grey", label=r"$\mu$" + f" = {np.nanmean(ratio_local_time):.2f}"
)
ax3.axhline(
    np.nanmean(ratio_local_time) + np.nanstd(ratio_local_time),
    color="grey",
    ls="dotted",
    label=r"$\sigma$" + f" = {np.nanstd(ratio_local_time):.3f}",
)
ax3.axhline(np.nanmean(ratio_local_time) - np.nanstd(ratio_local_time), color="grey", ls="dotted")

ax1.set_ylim(0, 0.08)
ax1.set_xlabel("Local Time [ hours ]\n(Binsize: 1)")
ax1.set_ylabel("Probability Density")
ax1.set_xticks(local_time_bins[::4])
ax1.set_title("Full Mission Distribution\n(Sampled Hourly)")

ax2.set_ylim(0, 0.08)
ax2.set_xlabel("Local Time [ hours ]\n(Binsize: 1)")
ax2.set_ylabel("Probability Density")
ax2.set_xticks(local_time_bins[::4])
ax2.set_title("Boundary Interval Distribution\n(Start Time)")

ax3.set_ylim(0.8, 1.2)
ax3.set_xlabel("Local Time [ hours ]\n(Binsize: 1)")
ax3.set_xticks(local_time_bins[::4])
ax3.set_ylabel("Ratio (b / a)")
ax3.set_title(f"Normalised Boundary Interval Distribution\n(Panels b / a)\nT-Test: p = {local_time_t_test.pvalue:.1e}")


for i, ax in enumerate(axes):
    ax.text(0.05, 0.95, axis_labels[i] + ")", transform=ax.transAxes)
    ax.margins(0)
    ax.legend()

plt.savefig("/home/daraghhollman/Main/mercury/Figures/philpott_local_time_distributions_extended.png", dpi=200)
