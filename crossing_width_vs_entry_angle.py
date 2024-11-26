"""
Script to plot the width of a BS crossing interval to the angle the entry trajectory makes with the surface normal
"""

import multiprocessing
import functools

import hermpy.boundary_crossings as boundaries
import hermpy.trajectory as traj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import scipy.stats

# Load the crossing intervals
crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/Work/mercury/DataSets/philpott_2020.xlsx"
)

# Limit only to bow shock crossings
crossings = crossings.loc[
    (crossings["Type"] == "BS_OUT") | (crossings["Type"] == "BS_IN")
]

# Get the time difference between the start and the stop
def Get_Crossing_Width(row):
    return (row["End Time"] - row["Start Time"]).total_seconds() / 60 # minutes


crossings["dt"] = crossings.apply(Get_Crossing_Width, axis=1)

# Functions to use pandas' apply with parallelisation
def Parallelise(data, function, number_of_processes):
    data_splits = np.array_split(data, number_of_processes)

    with multiprocessing.Pool(number_of_processes) as pool:

        data = pd.concat(pool.map(function, data_splits))

    return data


def Apply_Function_To_Subset(function, data_subset):
    return data_subset.apply(function, axis=1)


def Parallelised_Apply(data, function, number_of_processes):
    return Parallelise(data, functools.partial(Apply_Function_To_Subset, function), number_of_processes)


# Get the angle between the trajectory and the bs normal
crossings["grazing angle"] = Parallelised_Apply(crossings, traj.Get_Grazing_Angle, 16)
# non-parallel version
# crossings["grazing angle"] = crossings.apply(traj.Get_Grazing_Angle, axis=1)

# Remove outliers
# crossings = crossings[np.abs(scipy.stats.zscore(crossings["dt"])) < 3]

# correlation = scipy.stats.pearsonr(crossings["grazing angle"], crossings["dt"])



fig, image_axis = plt.subplots()


heatmap, x_edges, y_edges = np.histogram2d(crossings["grazing angle"], crossings["dt"], bins=30)
extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

image = image_axis.imshow(heatmap.T, extent=extent, origin="lower", norm="log", aspect="auto")

ax1_divider = make_axes_locatable(image_axis)
cax = ax1_divider.append_axes("right", size="5%", pad="2%")

fig.colorbar(image, cax=cax, label="Num. Crossings")

image_axis.set_ylabel("Crossing Interval Length [minutes]")
image_axis.set_xlabel("Grazing Angle [deg.]")

image_axis.margins(0)


plt.show()


"""
fig, (image_axis, line_axis) = plt.subplots(2, 1, height_ratios=[3, 1], sharex=True)


heatmap, x_edges, y_edges = np.histogram2d(crossings["grazing angle"], crossings["dt"], bins=20)
extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

image = image_axis.imshow(heatmap.T, extent=extent, origin="lower", norm="log", aspect="auto")

ax1_divider = make_axes_locatable(image_axis)
cax = ax1_divider.append_axes("right", size="5%", pad="2%")
ax2_divider = make_axes_locatable(line_axis)
_ = ax2_divider.append_axes("right", size="5%", pad="2%")
_.axis("off")

fig.colorbar(image, cax=cax, label="Num. Crossings")

image_axis.set_ylabel("Crossing Interval Length [minutes]")
line_axis.set_ylabel("Column Standard Deviation")
line_axis.set_xlabel("Grazing Angle [deg.]")

image_axis.margins(0)
line_axis.margins(0)


### Standard deviation of columns as a function of grazing angle
standard_deviations = scipy.stats.std(heatmap, axis=0)

x_centres = (x_edges[:-1] + x_edges[1:]) / 2
line_axis.plot(x_centres, standard_deviations)


plt.show()
"""
