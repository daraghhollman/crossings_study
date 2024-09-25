"""
Script to create histograms depicting the distribution of crossing intervals in heliocentric distance and local time.

For each crossing interval, we determine the midpoint and calculate Mercury's distance to the sun and the local time of the spacecraft at that time.
"""

import datetime as dt

import hermpy.boundary_crossings as boundaries
import hermpy.trajectory as trajectory
import hermpy.plotting_tools as plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from tqdm import tqdm

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

# Make histograms
fig, axes = plt.subplots(1, 2, sharey=True)

ax1, ax2 = axes

heliocentric_bins = np.arange(0.75, 1.25 + 0.01, 0.01)
local_time_bins = np.arange(0, 24 + 1, 1)

ax1.hist(
    heliocentric_distances,
    bins=heliocentric_bins,
    label=f"{len(crossings)} crossing intervals",
    color="indianred",
)
ax2.hist(
    local_times,
    bins=local_time_bins,
    label=f"{len(crossings)} crossing intervals",
    color="indianred",
)

ax1.set_xlabel("Heliocentric Distance [ semi-major axes ]\n(Binsize: 0.01)")
ax2.set_xlabel("Local Time [ hours ]\n(Binsize: 1)")

ax2.set_xticks(local_time_bins[::4])

ax1.set_ylabel("Number of Boundary Intervals")

for ax in axes:
    ax.margins(0)
    ax.legend()

fig.suptitle("Philpott Boundary Interval Distributions")
plt.show()
