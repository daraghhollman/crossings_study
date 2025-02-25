"""
Script to load the boundary normal calculations from crossings_study/add_grazing_angles.py,
and plot the output normal vectors to verify them.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hermpy import plotting, trajectory, utils

crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/philpott_grazing_angles.csv",
    parse_dates=["Start Time", "End Time"],
).iloc[0:200]
crossings["Normal Vector"] = crossings["Normal Vector"].apply(
    lambda x: list(map(float, x.strip("[]").split()))
)

# Get positions
middle_times = crossings["Start Time"] + (
    crossings["End Time"] - crossings["Start Time"]
)

positions = trajectory.Get_Position(
    "MESSENGER", middle_times, frame="MSM", aberrate=True
)
positions /= utils.Constants.MERCURY_RADIUS_KM

fig, ax = plt.subplots()

for (i, crossing), position in zip(crossings.iterrows(), positions):

    if "BS" in crossing["Type"]:
        colour = "black"

    else:
        colour = "indianred"

    ax.scatter(position[0], np.sqrt(position[1] ** 2 + position[2] ** 2), color=colour)

    vector_length = 1
    ax.arrow(
        position[0],
        np.sqrt(position[1] ** 2 + position[2] ** 2),
        vector_length * crossing["Normal Vector"][0],
        vector_length * crossing["Normal Vector"][1],
        width=0.01,
        head_width=0.1,
        head_length=0.1,
        ec="grey",
        fc="grey",
        alpha=0.3,
        zorder=5,
    )


plotting.Format_Cylindrical_Plot(ax)
plotting.Plot_Magnetospheric_Boundaries(ax)

ax.legend()

plt.show()
