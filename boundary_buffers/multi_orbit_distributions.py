import datetime as dt
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import spiceypy as spice

from hermpy import boundary_crossings, mag, plotting_tools, trajectory

mpl.rcParams["font.size"] = 12

############################################################
#################### LOADING FILES #########################
############################################################

root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"

metakernel = "/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt"
spice.furnsh(metakernel)

philpott_crossings = boundary_crossings.Load_Crossings(
    "/home/daraghhollman/Main/mercury/philpott_2020_reformatted.csv"
)

# Define the time segement we want to look at
# Specifically, the time segmenet of the centre orbit.
# We can convert this to minutes before / after apoapsis
# to look at the same time in other orbits
start = dt.datetime(year=2013, month=4, day=29, hour=0, minute=50)
end = dt.datetime(year=2013, month=4, day=29, hour=1, minute=10)
data_length = end - start

# Determine the file padding needed to display all the orbits wanted
search_start = start - dt.timedelta(days=1)
search_end = end + dt.timedelta(days=1)

data = mag.Load_Between_Dates(root_dir, search_start, search_end)

#############################################################
##################### FINDING ORBITS ########################
#############################################################

# Set the number of orbits either side
number_of_orbits = 3
approx_orbit_period = dt.timedelta(hours=12)

apoapsis_altitudes, apoapsis_times = trajectory.Get_All_Apoapsis_In_Range(
    start - number_of_orbits * approx_orbit_period,
    start + number_of_orbits * approx_orbit_period,
    number_of_orbits_to_include=5,
)

# Determine how far before apoapsis our start time is.
middle_index = len(apoapsis_times) // 2

middle_apoapsis_time = apoapsis_times[middle_index]
middle_apoapsis_altitude = apoapsis_altitudes[middle_index]


middle_data = mag.Strip_Data(data, start, end)
# Converting to MSM
middle_data = mag.MSO_TO_MSM(middle_data)
# Accounting for solar wind aberration angle
middle_data = mag.Adjust_For_Aberration(middle_data)

# Create new column in data for minutes before apoapsis
minutes_before_apoapsis = []

for date in middle_data["date"]:
    minutes_before_apoapsis.append((middle_apoapsis_time - date).total_seconds() / 60)

middle_data["minutes before apoapsis"] = minutes_before_apoapsis

minutes_before = middle_data["minutes before apoapsis"][0]


data_groups = []
for apoapsis_time in apoapsis_times:

    start_time = apoapsis_time - dt.timedelta(minutes=minutes_before)
    end_time = start_time + data_length

    new_data = mag.Strip_Data(
        data,
        start_time,
        end_time,
    )
    # Converting to MSM
    new_data = mag.MSO_TO_MSM(new_data)
    # Accounting for solar wind aberration angle
    new_data = mag.Adjust_For_Aberration(new_data)

    new_data["minutes before apoapsis"] = minutes_before_apoapsis

    data_groups.append(new_data)

####################################################
################## PLOTTING MAG ####################
####################################################

fig = plt.figure(figsize=(12, 20))

ax1 = plt.subplot2grid((len(data_groups), 5), (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid((len(data_groups), 5), (3, 0), colspan=2, rowspan=2)
trajectory_axes = [ax1, ax2]


mag_axes: list = [0] * len(data_groups)
hist_axes: list = [0] * len(data_groups)
hist_twin_axes = []

for i in range(len(data_groups)):
    mag_axes[i] = plt.subplot2grid((len(data_groups), 5), (i, 2), colspan=2)
    hist_axes[i] = plt.subplot2grid((len(data_groups), 5), (i, 4), colspan=1)


# We need to get the max value for constant axis scaling accross the 5 plots
max_mag = 0
max_hist = 0

for i, orbit_data in enumerate(data_groups):

    colour = "black"

    if i == middle_index:
        label = f"{orbit_data['date'].iloc[0].strftime("%Y-%m-%d %H:%M:%S")} to\n{orbit_data['date'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")}"
    else:
        if (i - middle_index) < 0:
            label = f"{abs(i - middle_index)} orbit(s) before"
        else:
            label = f"{i - middle_index} orbit(s) after"

    # Plot the mag data
    mag_axes[i].plot(
        orbit_data["minutes before apoapsis"],
        orbit_data["mag_total"],
        color=colour,
        lw=0.8,
    )

    # Label the panel
    mag_axes[i].annotate(
        label,
        xy=(1, 1),
        xycoords="axes fraction",
        size=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", fc="w"),
    )

    # Add the boundary crossings
    boundary_crossings.Plot_Crossings_As_Minutes_Before(
        mag_axes[i],
        philpott_crossings,
        orbit_data["date"].iloc[0],
        orbit_data["date"].iloc[-1],
        apoapsis_times[i],
        show_partial_crossings=False,
    )

    # Format the panels
    if np.max(orbit_data["mag_total"]) > max_mag:
        max_mag = np.max(orbit_data["mag_total"])

    mag_axes[i].set_xlim(minutes_before, np.min(orbit_data["minutes before apoapsis"]))
    mag_axes[i].set_ylim(0, max_mag)

    mag_axes[i].set_xmargin(0)
    mag_axes[i].tick_params("x", which="major", direction="out", length=8, width=1)
    mag_axes[i].tick_params("x", which="minor", direction="out", length=4, width=0.8)
    mag_axes[i].tick_params("y", which="major", direction="out", length=8, width=1)
    mag_axes[i].tick_params("y", which="minor", direction="out", length=4, width=0.8)

    mag_axes[i].xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    mag_axes[i].yaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    if mag_axes[i] != mag_axes[-1]:
        mag_axes[i].set_xticklabels([])

    # In addition to this, we want to sample the distributions of data before and after each boundary.
    # We first find the time of the boundary crossing within the orbit data.
    current_crossing = philpott_crossings[
        (philpott_crossings["start"] > orbit_data.iloc[0]["date"])
        & (philpott_crossings["end"] < orbit_data.iloc[-1]["date"])
    ]
    if len(current_crossing) > 1:
        raise Exception("There is more than one crossing within the data plotted")
    else:
        current_crossing = current_crossing.iloc[0]

    # We then want to sample at some buffer before the boundary, for some length of time
    # and plot the distribution.
    # We do the same for afterwards, with a different buffer and length of time.
    buffer_before = 5  # minutes
    length_before = 10  # minutes
    buffer_after = 5  # minutes
    length_after = 10  # minutes

    # We can simply load the data and re-strip, this is fast enough.
    pre_slice_start = current_crossing["start"] - dt.timedelta(
        minutes=(buffer_before + length_before)
    )
    pre_slice_end = current_crossing["start"] - dt.timedelta(minutes=buffer_before)

    post_slice_start = current_crossing["end"] + dt.timedelta(minutes=buffer_after)
    post_slice_end = current_crossing["end"] + dt.timedelta(
        minutes=(buffer_after + length_after)
    )

    data_sample_before = mag.Strip_Data(data, pre_slice_start, pre_slice_end)
    data_sample_after = mag.Strip_Data(data, post_slice_start, post_slice_end)


    # Add histogram data
    bin_step = 1

    hist_data_before, _, _ = hist_axes[i].hist(
        data_sample_before["mag_total"],
        np.arange(
            np.min(data_sample_before["mag_total"]),
            np.max(data_sample_before["mag_total"]),
            bin_step,
        ),
        color="red",
        orientation="horizontal",
        density=True,
        label="Magnetosheath",
        alpha=0.7
    )

    hist_twin_axes.append(hist_axes[i].twiny())

    hist_data_after, _, _ = hist_twin_axes[i].hist(
        data_sample_after["mag_total"],
        np.arange(
            np.min(data_sample_after["mag_total"]),
            np.max(data_sample_after["mag_total"]),
            bin_step,
        ),
        color="blue",
        orientation="horizontal",
        density=True,
        label="Solar Wind",
        alpha=0.7
    )

for ax in hist_axes:
    ax.set_xticks([])

    ax.set_ylim(0, max_mag)
    ax.set_yticklabels([])

    ax.set_xmargin(0)
    ax.tick_params("y", which="major", direction="out", length=8, width=1)
    ax.tick_params("y", which="minor", direction="out", length=4, width=0.8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))

for ax in hist_twin_axes:
    ax.set_xticks([])

    ax.set_ylim(0, max_mag)
    ax.set_yticklabels([])

    ax.set_xmargin(0)
    ax.tick_params("y", which="major", direction="out", length=8, width=1)
    ax.tick_params("y", which="minor", direction="out", length=4, width=0.8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))

mag_axes[-1].set_xlabel("Minutes before apoapsis")
mag_axes[middle_index].set_ylabel("|B| [nT]")

fig.text(0.95, 0.5, f"Distributions of |B| sampled for {length_before} minutes,\nwith a {buffer_before} minute buffer on the boundary intervals\n\n( Magnetosheath (red), Solar Wind (blue) )", ha='center', va='center', rotation=270)

# fig.text(0.06, 0.35, "|B| [nT]", ha="center", va="center", rotation="vertical")

#################################################################
################### PLOTING TRAJECTORIES ########################
#################################################################

# Here we just plot the trajectory of the middle orbit, along with some padding

frame = "MSM"

time_padding = dt.timedelta(hours=24)

start = data_groups[middle_index]["date"].iloc[0]
end = data_groups[middle_index]["date"].iloc[-1]

dates = [start, end]

padded_dates = [
    (start - time_padding),
    (end + time_padding),
]

positions = trajectory.Get_Trajectory("Messenger", dates, frame=frame, aberrate=True)
padded_positions = trajectory.Get_Trajectory(
    "Messenger", padded_dates, frame=frame, aberrate=True
)

# Convert from km to Mercury radii
positions /= 2439.7
padded_positions /= 2439.7


trajectory_axes[0].plot(
    positions[:, 0],
    positions[:, 1],
    color="magenta",
    lw=3,
    zorder=10,
    label="Plotted Trajectory",
)
trajectory_axes[1].plot(
    positions[:, 0],
    positions[:, 2],
    color="magenta",
    lw=3,
    zorder=10,
)
trajectory_axes[0].plot(
    padded_positions[:, 0],
    padded_positions[:, 1],
    color="grey",
    label=r"Trajectory $\pm$ "
    + str(int(time_padding.total_seconds() / 3600))
    + " hours",
)
trajectory_axes[1].plot(
    padded_positions[:, 0],
    padded_positions[:, 2],
    color="grey",
)

planes = ["xy", "xz"]
shaded = ["left", "left"]
for i, ax in enumerate(trajectory_axes):
    plotting_tools.Plot_Mercury(
        ax, shaded_hemisphere=shaded[i], plane=planes[i], frame=frame
    )
    plotting_tools.Add_Labels(ax, planes[i], frame=frame, aberrate=True)
    plotting_tools.Plot_Magnetospheric_Boundaries(ax, plane=planes[i], add_legend=True)
    plotting_tools.Square_Axes(ax, 4)

"""
trajectory_axes[0].legend(
    bbox_to_anchor=(0.5, 1.2), loc="center", ncol=2, borderaxespad=0.5
)
"""

# plt.savefig(f"./multi_orbit_plot_{start}")
plt.show()
