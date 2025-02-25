"""
Script to load the grazing angle calculations from crossings_study/add_grazing_angles.py,
and plot the average grazing angle in space.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hermpy import plotting, trajectory, utils

crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/philpott_grazing_angles.csv",
    parse_dates=["Start Time", "End Time"],
)

# Get positions
middle_times = crossings["Start Time"] + (
    crossings["End Time"] - crossings["Start Time"]
)

positions = trajectory.Get_Position(
    "MESSENGER", middle_times, frame="MSM", aberrate=False
)
positions /= utils.Constants.MERCURY_RADIUS_KM

fig, ax = plt.subplots()

box_size = 20
x_bins = np.linspace(np.min(positions[:, 0]), np.max(positions[:, 0]), box_size)
rho_bins = np.linspace(
    0, np.max(np.array(np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2))), box_size
)

# Find the average in each bin
sum_values, _, _ = np.histogram2d(
    positions[:, 0],
    np.sqrt(
        positions[:, 1] ** 2 + positions[:, 2] ** 2,
    ),
    bins=[x_bins, rho_bins],
    weights=crossings["Grazing Angle (deg.)"],
)

counts, _, _ = np.histogram2d(
    positions[:, 0],
    np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2),
    bins=[x_bins, rho_bins],
)

average_grazing_angles = np.divide(sum_values, counts, where=counts > 0)

plt.pcolormesh(x_bins, rho_bins, average_grazing_angles.T)


plotting.Format_Cylindrical_Plot(ax)
plotting.Plot_Magnetospheric_Boundaries(ax, zorder=5)

ax.set_xlim(np.min(positions[:, 0]), np.max(positions[:, 0]))
ax.set_ylim(0, np.max(np.array(np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2))))

plt.colorbar(label="Average Grazing Angle (deg.)")

plt.show()
