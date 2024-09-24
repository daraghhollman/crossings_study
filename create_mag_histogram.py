"""
A script to reproduce the bimodal histogram plots from Adam Healy's slides

The aim is to input any two times and create the plot between those times.
"""

import datetime as dt
from glob import glob

import hermpy.boundary_crossings as boundaries
import hermpy.mag as mag
import hermpy.plotting_tools as plotting
import hermpy.trajectory as trajectory
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit

# PARAMETERS
mpl.rcParams["font.size"] = 14
root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"
metakernel = "/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt"
philpott_crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/mercury/philpott_crossings.p"
)

start_time = dt.datetime(year=2011, month=6, day=5, hour=22, minute=5)
end_time = dt.datetime(year=2011, month=6, day=5, hour=23, minute=25)


# STEP ONE: LOAD DATA
if (end_time - start_time).days == 0:
    dates_to_load = [start_time]

else:
    dates_to_load: list[dt.datetime] = [
        start_time + dt.timedelta(days=i) for i in range((end_time - start_time).days)
    ]

files_to_load: list[str] = []
for date in dates_to_load:
    file: list[str] = glob(
        root_dir
        + f"{date.strftime('%Y')}/*/MAGMSOSCIAVG{date.strftime('%y%j')}_01_V08.TAB"
    )

    if len(file) > 1:
        raise ValueError("ERROR: There are duplicate data files being loaded.")
    elif len(file) == 0:
        raise ValueError("ERROR: The data trying to be loaded doesn't exist!")

    files_to_load.append(file[0])

data = mag.Load_Messenger(files_to_load)


# STEP TWO: MAKE DATA ADJUSTMENTS

# Shortening to only the times we need to plot
data = mag.Strip_Data(data, start_time, end_time)

# Converting to MSM'
data = mag.MSO_TO_MSM(data)
data = mag.Adjust_For_Aberration(data)


# STEP THREE: PLOTTING TIME SERIES

fig = plt.figure()

# Make trajectory axes
ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=1, rowspan=2)
ax2 = plt.subplot2grid((4, 2), (0, 1), colspan=1, rowspan=2)
trajectory_axes = [ax1, ax2]

ax3 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
ax4 = plt.subplot2grid((4, 2), (3, 0), colspan=2)
mag_axes = [ax3, ax4]


# Plot MAG
to_plot = ["mag_x", "mag_total"]
y_labels = ["B$_x$", "|B|"]
for i, ax in enumerate(mag_axes):

    ax.plot(data["date"], data[to_plot[i]], color="black", lw=0.8)
    ax.set_ylabel(y_labels[i])

    # Plot hline at 0
    ax.axhline(0, color="grey", ls="dotted")

    ax.set_xmargin(0)
    ax.tick_params("x", which="major", direction="inout", length=16, width=1)
    ax.tick_params("x", which="minor", direction="inout", length=8, width=0.8)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))


# Add ephemeris
plotting.Add_Tick_Ephemeris(
    mag_axes[-1],
    metakernel,
    include={"date", "hours", "minutes", "range", "local time"},
)


# Get trajectory data from spice
time_padding = dt.timedelta(hours=6)
spice_dates = [
    start_time.strftime("%Y-%m-%d %H:%M:%S"),
    end_time.strftime("%Y-%m-%d %H:%M:%S"),
]

padded_dates = [
    (start_time - time_padding).strftime("%Y-%m-%d %H:%M:%S"),
    (end_time + time_padding).strftime("%Y-%m-%d %H:%M:%S"),
]

frame = "MSM"

# Get positions in MSO coordinate system
positions = trajectory.Get_Trajectory(
    "Messenger", spice_dates, metakernel, frame=frame, aberrate=True
)
padded_positions = trajectory.Get_Trajectory(
    "Messenger", padded_dates, metakernel, frame=frame, aberrate=True
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
    plotting.AddLabels(ax, planes[i], frame=frame, aberrate=True)
    plotting.PlotMagnetosphericBoundaries(ax, plane=planes[i], add_legend=True)
    plotting.SquareAxes(ax, 4)

trajectory_axes[1].legend(
    bbox_to_anchor=(0.5, 1.2), loc="center", ncol=2, borderaxespad=0.5
)

plt.show()


# STEP FOUR: PLOTTING HISTOGRAM OF B_X

fig, ax = plt.subplots()

binsize = 1
bins = np.arange(np.min(data["mag_x"]), np.max(data["mag_x"]), binsize)
hist_data, bin_edges, _ = ax.hist(
    data["mag_x"],
    bins=bins,
    density=True,
    color="cornflowerblue",
    label=f"{start_time.strftime('%Y-%M-%d %H:%M')} to {start_time.strftime('%Y-%M-%d %H:%M')}",
)

bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2


# Get double gaussian fit
def Double_Gaussian(x, c1, mu1, sigma1, c2, mu2, sigma2):
    res = c1 * np.exp(-((x - mu1) ** 2.0) / (2.0 * sigma1**2.0)) + c2 * np.exp(
        -((x - mu2) ** 2.0) / (2.0 * sigma2**2.0)
    )
    return res


pars, cov = curve_fit(
    Double_Gaussian, bin_centres, hist_data, [0.01, -20, 10, 0.01, 20, 10]
)

ax.plot(
    bins,
    Double_Gaussian(
        bins,
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
        pars[5],
    ),
    color="indianred",
    lw=3,
    label="Double Gaussian Fit",
)

# WE CAN FIND THE SADDLE POINT BY GETTING THE PEAK OF THE NEGATIVE GAUSSIAN


ax.set_ylabel("Probability Density of Measurements")
ax.set_xlabel(r"B$_x$ " + f" (binsize {binsize} nT)")

plt.legend()
plt.show()
