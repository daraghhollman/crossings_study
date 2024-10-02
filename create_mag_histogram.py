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
import scipy.signal
import spiceypy as spice
from scipy.optimize import curve_fit

import bimodal_tools
from plotting_tools import colored_line

histogram_parameter = "magnitude"  # options: magnitude, x
fit_curve = True
split_method = "minimum_point"  # none, threshold, midpoint, minimum_point

# if threshold is selected:
number_of_sigmas = 3


# PARAMETERS
mpl.rcParams["font.size"] = 15
root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"
metakernel = "/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt"
spice.furnsh(metakernel)
philpott_crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/mercury/philpott_2020_reformatted.csv"
)

start_time = dt.datetime(year=2012, month=7, day=2, hour=17, minute=20)
end_time = dt.datetime(year=2012, month=7, day=2, hour=17, minute=40)

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

    # Plot hline at 0
    # ax.axhline(0, color="grey", ls="dotted")

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


# 4: PLOTTING HISTOGRAM OF B_X
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
    label=f"{start_time.strftime('%Y-%M-%d %H:%M:%S')} to {end_time.strftime('%Y-%M-%d %H:%M:%S')}"
    + "\nsampled per second",
)

bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2


# Get double gaussian fit
def Double_Gaussian(x, c1, mu1, sigma1, c2, mu2, sigma2):
    res = c1 * np.exp(-((x - mu1) ** 2.0) / (2.0 * sigma1**2.0)) + c2 * np.exp(
        -((x - mu2) ** 2.0) / (2.0 * sigma2**2.0)
    )
    return res


if fit_curve:
    curve_fit_guess_params = [
        0.01,
        np.mean(histogram_parameter_values) - np.std(histogram_parameter_values),
        5,
        0.01,
        np.mean(histogram_parameter_values) + np.std(histogram_parameter_values),
        5,
    ]
    pars, cov = curve_fit(
        # We need to pass some initial guess parameters, these were chosen arbitraraly
        Double_Gaussian,
        bin_centres,
        hist_data,
        curve_fit_guess_params,
    )

    fit_range = np.linspace(bins[0], bins[-1], 100)
    fit_values = Double_Gaussian(
        fit_range,
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
        pars[5],
    )

    histogram_axis.plot(
        fit_range,
        fit_values,
        color="magenta",
        lw=3,
        label="Double Gaussian Fit",
    )

    match split_method:
        case "none":
            pass

        case "threshold":
            # Upper threshold:
            #       If the data is low, and we go higher than this then we switch.
            #       Upper threshold is the mean - 1 sigma of the upper gaussian
            upper_threshold = pars[4] - number_of_sigmas * pars[5]

            # Lower threshold:
            #       If the data is high, and we go lower than this then we switch.
            #       Lower threshold is the mean + 1 sigma of the lower gaussian
            lower_threshold = pars[1] + number_of_sigmas * pars[2]

            region_index = []
            # Low region = 0, high region = 1
            # We determine the initial region from a comparison of the first data point
            if histogram_parameter_values[0] > upper_threshold:
                current_region = 1
            elif histogram_parameter_values[0] < lower_threshold:
                current_region = 0
            # We need to handle cases between
            else:
                upper_difference = upper_threshold - histogram_parameter_values
                lower_difference = histogram_parameter_values - lower_threshold

                if upper_difference >= lower_difference:
                    current_region = 1
                else:
                    current_region = 0

            for i, field_value in enumerate(histogram_parameter_values):

                if field_value > upper_threshold:
                    current_region = 1
                elif field_value < lower_threshold:
                    current_region = 0

                region_index.append(current_region)

            colored_line(
                data["date"], data["mag_x"], region_index, ax=mag_axes[0], cmap="bwr"
            )
            colored_line(
                data["date"],
                data["mag_total"],
                region_index,
                ax=mag_axes[1],
                cmap="bwr",
            )

            histogram_axis.axvline(
                upper_threshold, color="red", label="Upper Threshold"
            )
            histogram_axis.axvline(
                lower_threshold, color="blue", label="Lower Threshold"
            )

        case "midpoint":
            # Find the midpoint between the two peaks
            midpoint = (pars[1] + pars[4]) / 2

            region_index = []

            for i, field_value in enumerate(histogram_parameter_values):

                if field_value > midpoint:
                    current_region = 1
                else:
                    current_region = 0

                region_index.append(current_region)

            colored_line(
                data["date"], data["mag_x"], region_index, ax=mag_axes[0], cmap="bwr"
            )
            colored_line(
                data["date"],
                data["mag_total"],
                region_index,
                ax=mag_axes[1],
                cmap="bwr",
            )

            histogram_axis.axvline(midpoint, color="orange", label="Midpoint")

        case "minimum_point":
            # Split based off of the minimum point of the gaussian distribution
            distribution_minimum_index, _ = scipy.signal.find_peaks(-fit_values)

            distribution_minimum = fit_range[distribution_minimum_index]

            region_index = []

            for i, field_value in enumerate(histogram_parameter_values):

                if field_value > distribution_minimum:
                    current_region = 1
                else:
                    current_region = 0

                region_index.append(current_region)

            colored_line(
                data["date"], data["mag_x"], region_index, ax=mag_axes[0], cmap="bwr"
            )
            colored_line(
                data["date"],
                data["mag_total"],
                region_index,
                ax=mag_axes[1],
                cmap="bwr",
            )

            histogram_axis.axvline(
                distribution_minimum, color="orange", label="Distribution Minimum"
            )


histogram_axis.set_ylabel("Probability Density of Measurements")
histogram_axis.set_xlabel(histogram_axis_label + f" (binsize {binsize} nT)")
histogram_axis.yaxis.tick_right()
histogram_axis.yaxis.set_label_position("right")

plt.legend()

plt.savefig(
    f"/home/daraghhollman/Main/mercury/Figures/bimodal/bimodal_{start_time.strftime("%Y_%m_%d__%H_%M_%S")}_{end_time.strftime("%Y_%m_%d__%H_%M_%S")}.png",
    dpi=300,
)
