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
    "/home/daraghhollman/Main/mercury/philpott_crossings.p"
)

crossings = crossings.iloc[50:100]

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
    position_start = np.array(
        [
            crossings.iloc[i]["start_x_msm"],
            crossings.iloc[i]["start_y_msm"],
            crossings.iloc[i]["start_z_msm"],
        ]
    )
    position_end = np.array(
        [
            crossings.iloc[i]["end_x_msm"],
            crossings.iloc[i]["end_y_msm"],
            crossings.iloc[i]["end_z_msm"],
        ]
    )
    # Get midpoint position
    #position_b = (position_start + position_end) / 2
    position_b = position_start

    position = trajectory.Get_Position("MESSENGER", crossings.iloc[i]["start"])

    if position is None:
        continue

    pos_a.append(position)
    pos_b.append(position_b)


    longitude = np.arctan2(position[1], position[0]) * 180 / np.pi

    if longitude < 0:
        longitude += 360

    local_times[i] = ((longitude + 180) * 24 / 360) % 24


pos_a = np.array(pos_a) / 2440
pos_b = np.array(pos_b)
fig, ax = plt.subplots()
ax.scatter(pos_a[:, 0], pos_a[:, 1], color="blue", label="SPICE Locations")
ax.scatter(pos_b[:, 0], pos_b[:, 1], color="red", label="Philpott Locations")
ax.plot(pos_a[:, 0], pos_a[:, 1], color="blue", alpha=0.4)
ax.plot(pos_b[:, 0], pos_b[:, 1], color="red", alpha=0.4)
plotting.Plot_Mercury(ax, shaded_hemisphere="left")
ax.set_aspect("equal")
ax.set_xlabel("X MSM [R$_M$]")
ax.set_ylabel("Y MSM [R$_M$]")
ax.legend()
plt.show()
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
