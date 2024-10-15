"""
Script to make a single row mag plot followed by a plot of histograms for the interval region and increasing buffers of solar wind data
"""

import datetime as dt
import os
from math import ceil, floor

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.optimize
import scipy.signal
import scipy.stats
import spiceypy as spice
from hermpy import boundary_crossings, mag, plotting_tools, trajectory

mpl.rcParams["font.size"] = 8

colours = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

component = "mag_total"  # "mag_total", "mag_x", "mag_y", "mag_z"

save = True
allow_multiple_intervals = True

# Defining buffer lengths
sample_buffer = 0  # minutes
sample_length = 10  # minutes

match component:
    case "mag_total":
        mag_axis_label = "|B| [nT]"
        use_log = True

    case "mag_x":
        mag_axis_label = "Bx [nT]"
        use_log = False

    case "mag_y":
        mag_axis_label = "By [nT]"
        use_log = False

    case "mag_z":
        mag_axis_label = "Bz [nT]"
        use_log = False

############################################################
#################### LOADING FILES #########################
############################################################

root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"

metakernel = "/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt"
spice.furnsh(metakernel)

philpott_crossings = boundary_crossings.Load_Crossings(
    "/home/daraghhollman/Main/Work/mercury/philpott_2020_reformatted.csv"
)
# Define the time segement we want to look at
# Specifically, the time segmenet of the centre orbit.
# We can convert this to minutes before / after apoapsis
# to look at the same time in other orbits
start = dt.datetime(year=2014, month=7, day=12, hour=21, minute=55)
end = dt.datetime(year=2014, month=7, day=12, hour=22, minute=20)
data_length = end - start

data = mag.Load_Between_Dates(root_dir, start, end, strip=True)

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


####################################################
################## PLOTTING MAG ####################
####################################################

fig = plt.figure(figsize=(11.7, 8))


num_hists = 2
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
mag_ax.plot(data["date"], data["mag_total"], color=colour, lw=1, label="|B|")
mag_ax.plot(
    data["date"], data["mag_x"], color=colours[2], lw=0.8, alpha=0.8, label="Bx"
)
mag_ax.plot(
    data["date"], data["mag_y"], color=colours[0], lw=0.8, alpha=0.8, label="By"
)
mag_ax.plot(
    data["date"], data["mag_z"], color=colours[-1], lw=0.8, alpha=0.8, label="Bz"
)

mag_leg = mag_ax.legend(
    bbox_to_anchor=(0.5, 1.1), loc="center", ncol=4, borderaxespad=0.5
)

# set the linewidth of each legend object
for legobj in mag_leg.legend_handles:
    legobj.set_linewidth(2.0)

# Label the panel
"""
mag_ax.annotate(
    label,
    xy=(1, 1),
    xycoords="axes fraction",
    size=10,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round", fc="w"),
)
"""

# Add the boundary crossings
boundary_crossings.Plot_Crossing_Intervals(
    mag_ax,
    data["date"].iloc[0],
    data["date"].iloc[-1],
    philpott_crossings,
    color=colours[3],
    lw=1.5,
)

# Format the panels
mag_ax.set_xmargin(0)
plotting_tools.Add_Tick_Ephemeris(mag_ax)
mag_ax.set_ylabel(mag_axis_label)

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
    color="black",
    lw=3,
    zorder=10,
    label="Plotted Trajectory",
)
trajectory_axes[1].plot(
    positions[:, 0],
    positions[:, 2],
    color="black",
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

# Get position during crossing interval
interval_positions = trajectory.Get_Trajectory(
    "Messenger",
    [current_crossing["start"], current_crossing["end"]],
    frame=frame,
    aberrate=True,
)
interval_positions /= 2439.7

trajectory_axes[0].plot(
    interval_positions[:, 0],
    interval_positions[:, 1],
    color=colours[3],
    lw=5,
    zorder=15,
    solid_capstyle="butt",
)
trajectory_axes[1].plot(
    interval_positions[:, 0],
    interval_positions[:, 2],
    color=colours[3],
    lw=5,
    zorder=15,
    solid_capstyle="butt",
)

# Get sample area position
sample_positions = trajectory.Get_Trajectory(
    "Messenger",
    [
        current_crossing["end"],
        current_crossing["end"] + dt.timedelta(minutes=sample_length),
    ],
    frame=frame,
    aberrate=True,
)
sample_positions /= 2439.7

trajectory_axes[0].plot(
    sample_positions[:, 0],
    sample_positions[:, 1],
    color=colours[0],
    lw=5,
    zorder=15,
    solid_capstyle="butt",
)
trajectory_axes[1].plot(
    sample_positions[:, 0],
    sample_positions[:, 2],
    color=colours[0],
    lw=5,
    zorder=15,
    solid_capstyle="butt",
)

# We also want to include the region of the boundary interval, and the sampled region
trajectory_axes[0].plot()

planes = ["xy", "xz"]
shaded = ["left", "left"]
for i, ax in enumerate(trajectory_axes):
    plotting_tools.Plot_Mercury(
        ax, shaded_hemisphere=shaded[i], plane=planes[i], frame=frame
    )
    plotting_tools.Add_Labels(ax, planes[i], frame=frame, aberrate=True)
    plotting_tools.Plot_Magnetospheric_Boundaries(ax, plane=planes[i], add_legend=True)
    plotting_tools.Square_Axes(ax, 4)

    ax.set_ylim(-2, 3)
    ax.set_xlim(-2, 3)

# Get data only within the interval:
interval_data = mag.Load_Between_Dates(
    root_dir, current_crossing["start"], current_crossing["end"], strip=True
)

mag_ax.axvline(
    current_crossing["end"] + dt.timedelta(minutes=sample_length), color=colours[0]
)
mag_ax.axvspan(
    current_crossing["end"],
    current_crossing["end"] + dt.timedelta(minutes=sample_length),
    color=colours[0],
    alpha=0.08,
)


min_x_values = []
max_x_values = []

max_y_values = []


# We then want to sample at some buffer before or after the boundary, for some length of time
# and plot the distribution.

# We can simply load the data and re-strip, this is fast enough.
sample_start = current_crossing["end"] + dt.timedelta(minutes=sample_buffer)
sample_end = current_crossing["end"] + dt.timedelta(
    minutes=(sample_buffer + sample_length)
)

sample_data = mag.Load_Between_Dates(root_dir, sample_start, sample_end, strip=True)

log_bins = 10 ** (np.arange(0, 10, 0.05))
if use_log:
    bins = log_bins

else:
    bins = np.linspace(
        floor(np.min(sample_data[component])), ceil(np.max(sample_data[component])), 16
    )


for i in range(2):

    # First histogram is for the data within the boundary interval
    if i == 0:
        hist_data_interval, _, _ = hist_axes[i].hist(
            interval_data[component],
            bins=bins,
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

    sample_hist_data, bin_edges, _ = hist_axes[i].hist(
        sample_data[component],
        bins=bins,
        color=colours[0],
        density=True,
    )
    hist_axes[i].annotate(
        f"N={len(sample_data[component])}",
        xy=(0, 1),
        xycoords="axes fraction",
        size=10,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="w"),
    )

    mean = np.mean(sample_data[component])
    median = np.median(sample_data[component])
    std = np.std(sample_data[component])
    skew = scipy.stats.skew(sample_data[component])
    kurtosis = scipy.stats.kurtosis(sample_data[component])

    hist_axes[i].axvline(mean, color="black", label=f"Mean: {mean:.1f} nT")
    hist_axes[i].axvline(
        median, ls="dashed", color=colours[2], label=f"Median: {median:.1f} nT"
    )

    hist_axes[i].axvline(
        mean + std, ls="dashed", color="grey", label=f"SD: {std:.1f} nT"
    )
    hist_axes[i].axvline(mean - std, ls="dashed", color="grey")

    hist_axes[i].plot(np.NaN, np.NaN, "-", color="none", label=f"Skew: {skew:.1f}")
    hist_axes[i].plot(
        np.NaN, np.NaN, "-", color="none", label=f"Kurtosis: {kurtosis:.1f}"
    )

    hist_axes[i].legend(bbox_to_anchor=(0.8, 1), loc="upper left", ncol=1)

    min_x_values.append(np.min(sample_data[component]))
    max_x_values.append(np.max(sample_data[component]))

    max_y_values.append(np.max(sample_hist_data))


for i in range(len(hist_axes)):

    # Set bottom labels
    if i != 0:
        hist_axes[i].set_xlabel(f"{mag_axis_label}")
        hist_axes[i].set_title("Near Boundary\nSolar Wind Distribution")

        hist_axes[i].set_ylim(0, np.max(max_y_values))

    else:
        hist_axes[i].set_xlabel(mag_axis_label)
        hist_axes[i].set_title("Boundary Interval Distribution")

    hist_axes[i].set_xlim(
        np.min(min_x_values), np.max(max_x_values + list(interval_data[component]))
    )

    if use_log:
        hist_axes[i].set_xscale("log")


    # Check if there are no major ticks
    if hist_axes[i].xaxis.get_major_ticks() == None:
        hist_axes[i].xaxis.set_minor_formatter(ticker.ScalarFormatter())


if save:
    save_path = "/home/daraghhollman/Main/Work/mercury/Figures/noon_north_bowshock/Solar_Wind_Investigation/" + start.strftime("%Y%m%d_%H%M") + "/"

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    fig.savefig(
        save_path
        + f"sw_sample_bs_out_{component}.pdf"
    )
else:
    plt.show()
