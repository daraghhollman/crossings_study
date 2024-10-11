"""
Script to make a single row mag plot followed by a plot of histograms for the interval region and increasing buffers of solar wind data
"""

import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
start = dt.datetime(year=2014, month=1, day=21, hour=2, minute=15)
end = dt.datetime(year=2014, month=1, day=21, hour=2, minute=30)
data_length = end - start

data = mag.Load_Between_Dates(root_dir, start, end, strip=True)

####################################################
################## PLOTTING MAG ####################
####################################################

fig = plt.figure()


num_hists = 4
mag_ax = plt.subplot2grid((5, num_hists + 2), (0, 2), colspan=4, rowspan=2)

hist_axes = []
for i in range(num_hists):
    hist_axes.append(plt.subplot2grid((5, num_hists + 2), (3, i + 2), rowspan=2))

trajectory_axes = []
for i in [0, 3]:
    trajectory_axes.append(
        plt.subplot2grid((5, num_hists + 2), (i, 0), colspan=2, rowspan=2)
    )


# We need to get the max value for constant axis scaling accross the 5 plots
colour = "black"

label = f"{data['date'].iloc[0].strftime("%Y-%m-%d %H:%M:%S")} to\n{data['date'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")}"

# Plot the mag data
mag_ax.plot(
    data["date"],
    data["mag_total"],
    color=colour,
    lw=0.8,
)

# Label the panel
mag_ax.annotate(
    label,
    xy=(1, 1),
    xycoords="axes fraction",
    size=10,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round", fc="w"),
)

# Add the boundary crossings
boundary_crossings.Plot_Crossing_Intervals(
    mag_ax, data["date"].iloc[0], data["date"].iloc[-1], philpott_crossings
)

# Format the panels
mag_ax.set_xmargin(0)
plotting_tools.Add_Tick_Ephemeris(mag_ax)
mag_ax.set_ylabel("|B| [nT]")

#################################################################
################### PLOTING TRAJECTORIES ########################
#################################################################

# Here we just plot the trajectory of the middle orbit, along with some padding

frame = "MSM"

time_padding = dt.timedelta(hours=6)

start = data["date"].iloc[0]
end = data["date"].iloc[-1]

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

# Defining buffer lengths
sample_buffer = [0, 10, 20]  # minutes
sample_length = 10  # minutes

# We want to sample the distributions of data before and after each boundary.
# We first find the time of the boundary crossing within the orbit data.
current_crossing = philpott_crossings[
    (philpott_crossings["start"] > data.iloc[0]["date"])
    & (philpott_crossings["end"] < data.iloc[-1]["date"])
]
if len(current_crossing) > 1:
    raise Exception("There is more than one crossing within the data plotted")
else:
    current_crossing = current_crossing.iloc[0]

# Get data only within the interval:
interval_data = mag.Load_Between_Dates(
    root_dir, current_crossing["start"], current_crossing["end"], strip=True
)

for i in range(len(sample_buffer) + 1):

    # First histogram is for the data within the boundary interval
    if i == 0:
        hist_data_interval, _, _ = hist_axes[i].hist(
            interval_data["mag_total"],
            color="black",
            density=True,
            label=f"Boundary Interval",
            alpha=0.7,
        )

        hist_axes[i].annotate(
            f"N={len(interval_data)}",
            xy=(0, 1),
            xycoords="axes fraction",
            size=10,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", fc="w"),
        )

        continue

    # We then want to sample at some buffer before or after the boundary, for some length of time
    # and plot the distribution.

    # We can simply load the data and re-strip, this is fast enough.

    sample_start = current_crossing["end"] + dt.timedelta(minutes=sample_buffer[i - 1])
    sample_end = current_crossing["end"] + dt.timedelta(
        minutes=(sample_buffer[i - 1] + sample_length)
    )

    sample_data = mag.Load_Between_Dates(root_dir, sample_start, sample_end, strip=True)

    sample_hist_data, _, _ = hist_axes[i].hist(
        sample_data["mag_total"],
        color="blue",
        density=True,
        label=f"Solar Wind\n"
        + r"$\Delta t$"
        + f" = {sample_length} minutes\nN={len(sample_data)}",
    )


for i in range(len(hist_axes)):

    # Set bottom labels
    if i != 0:
        hist_axes[i].set_xlabel(f"|B| [nT]\n{sample_buffer[i-1]} minutes buffer")
    else:
        hist_axes[i].set_xlabel(f"|B| [nT]")


plt.show()
