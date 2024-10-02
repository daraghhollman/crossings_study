"""
A script to reproduce the bimodal histogram plots from Adam Healy's slides

The aim is to input any two times and create the plot between those times.
"""

import datetime as dt

import hermpy.boundary_crossings as boundaries
import hermpy.mag as mag
import hermpy.plotting_tools as plotting
import hermpy.trajectory as trajectory
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import spiceypy as spice

import bimodal_tools

histogram_parameter = "magnitude"  # options: magnitude, x
fit_curve = True
split_method = "likelihood"  # none, likelihood, threshold, midpoint, minimum_point


# PARAMETERS
mpl.rcParams["font.size"] = 15
root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"
metakernel = "/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt"
spice.furnsh(metakernel)
philpott_crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/mercury/philpott_2020_reformatted.csv"
)

start_time = dt.datetime(year=2011, month=3, day=31, hour=13, minute=35)
end_time = dt.datetime(year=2011, month=3, day=31, hour=14, minute=15)

# 1: LOAD DATA
data = mag.Load_Between_Dates(root_dir, start_time, end_time)

# 2: MAKE DATA ADJUSTMENTS
# Shortening to only the times we need to plot,
# changing from MSO to MSM, and aberrating the data.
data = mag.Strip_Data(data, start_time, end_time)
data = mag.MSO_TO_MSM(data)
data = mag.Adjust_For_Aberration(data)

# 3: PLOTTING TIME SERIES
# Set up figure
fig, trajectory_axes, mag_axes, histogram_axis = bimodal_tools.Create_Axes()

# Plot MAG
to_plot = ["mag_x", "mag_total"]
y_labels = ["B$_x$", "|B|"]
for i, ax in enumerate(mag_axes):

    if split_method == "None":
        # If we're not dividing the data, we simply plot the data as is
        ax.plot(
            data["date"], data[to_plot[i]], color="black", lw=0.8, label="MESSENGER MAG"
        )
    else:
        ax.plot(data["date"], data[to_plot[i]], color="black", lw=0.8, zorder=-1)

    boundaries.Plot_Crossing_Intervals(ax, start_time, end_time, philpott_crossings)
    ax.set_ylabel(y_labels[i])

    ax.set_xmargin(0)
    ax.tick_params("x", which="major", direction="inout", length=16, width=1)
    ax.tick_params("x", which="minor", direction="inout", length=8, width=0.8)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))


# Add ephemeris
plotting.Add_Tick_Ephemeris(
    mag_axes[-1],
    include={"hours", "minutes", "range", "local time", "latitude"},
)
mag_axes[0].set_xticklabels([])


# 4: LOADING TRAJECTORY DATA
# Get trajectory data from spice
time_padding = dt.timedelta(hours=6)
spice_dates = [
    start_time,
    end_time,
]

padded_dates = [
    (start_time - time_padding),
    (end_time + time_padding),
]

frame = "MSM"

# Get positions in MSO coordinate system
positions = trajectory.Get_Trajectory(
    "Messenger",
    spice_dates,
    steps=int((end_time - start_time).total_seconds()) + 1,
    frame=frame,
    aberrate=True,
)
padded_positions = trajectory.Get_Trajectory(
    "Messenger", padded_dates, frame=frame, aberrate=True
)

# Convert from km to Mercury radii
positions /= 2439.7
padded_positions /= 2439.7

# 5: PLOT TRAJECTORIES
trajectory_axes[0].plot(
    positions[:, 0], positions[:, 1], color="magenta", lw=3, zorder=10
)
trajectory_axes[1].plot(
    positions[:, 0],
    positions[:, 2],
    color="magenta",
    lw=3,
    zorder=10,
    label="Plotted Trajectory",
)

# We also would like to give context and plot the orbit around this
trajectory_axes[0].plot(padded_positions[:, 0], padded_positions[:, 1], color="grey")
trajectory_axes[1].plot(
    padded_positions[:, 0],
    padded_positions[:, 2],
    color="grey",
    label=r"Trajectory $\pm$ 6 hours",
)

planes = ["xy", "xz"]
for i, ax in enumerate(trajectory_axes):
    plotting.Plot_Mercury(ax, shaded_hemisphere="left", plane=planes[i], frame=frame)
    plotting.Add_Labels(ax, planes[i], frame=frame, aberrate=True)
    plotting.Plot_Magnetospheric_Boundaries(ax, plane=planes[i], add_legend=True)
    plotting.Square_Axes(ax, 4)

trajectory_axes[1].legend(
    bbox_to_anchor=(-0.1, 1.2), loc="center", ncol=2, borderaxespad=0.5
)


# 5: PLOTTING BIMODAL HISTOGRAM
match histogram_parameter:
    case "x":
        histogram_parameter_values = data["mag_x"]
        histogram_axis_label = r"B$_x$"
    case "magnitude":
        histogram_parameter_values = np.sqrt(
            data["mag_x"] ** 2 + data["mag_y"] ** 2 + data["mag_z"] ** 2
        )
        histogram_axis_label = r"|B|"
    case _:
        raise ValueError("Histogram Parameter is not set!")

binsize = 1  # nT
bins = np.arange(
    np.min(histogram_parameter_values), np.max(histogram_parameter_values), binsize
)
hist_data, bin_edges, _ = histogram_axis.hist(
    histogram_parameter_values,
    bins=bins,
    density=True,
    color="black",
    label=f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
    + "\nsampled per second",
)

histogram_axis.set_ylabel("Probability Density of Measurements")
histogram_axis.set_xlabel(histogram_axis_label + f" (binsize {binsize} nT)")
histogram_axis.yaxis.tick_right()
histogram_axis.yaxis.set_label_position("right")



# 6: FITTING A DOUBLE GAUSSIAN TO THE DISTRIBUTION
if fit_curve:
    bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2

    curve_fit_guess_params = [
        0.01,
        np.mean(histogram_parameter_values) - np.std(histogram_parameter_values),
        5,
        0.01,
        np.mean(histogram_parameter_values) + np.std(histogram_parameter_values),
        5,
    ]

    population_fit = bimodal_tools.Population_Fit(bin_centres, hist_data, curve_fit_guess_params)

    histogram_axis.plot(
        population_fit.x_range,
        population_fit.output_y_values,
        color="grey",
        lw=10,
        alpha=0.5,
        zorder=1,
        label="Double Gaussian Fit",
    )
    histogram_axis.plot(
        population_fit.x_range,
        population_fit.population_a,
        color="blue",
        lw=3,
        zorder=5,
        label="Population A",
    )
    histogram_axis.plot(
        population_fit.x_range,
        population_fit.population_b,
        color="red",
        lw=3,
        zorder=5,
        label="Population B",
    )


    bimodal_tools.Split_Distribution(
        data,
        histogram_parameter_values,
        population_fit,
        fig,
        mag_axes,
        histogram_axis,
        method=split_method,
    )
    histogram_axis.legend()


plt.savefig(
    f"/home/daraghhollman/Main/mercury/Figures/bimodal/bimodal_{start_time.strftime("%Y_%m_%d__%H_%M_%S")}_{end_time.strftime("%Y_%m_%d__%H_%M_%S")}.png",
    dpi=300,
)
