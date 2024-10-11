"""
A file to create distributions of the lengths of time between each crossing interval in:
    - Solar Wind
    - Magnetosphere
    - Magnetosheath

With the aim being to find a maxiumum time difference to buffer the boundaries by
"""

import hermpy.boundary_crossings as boundaries
import hermpy.plotting_tools as hermplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from numpy import number

mpl.rcParams["font.size"] = 12

crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/mercury/philpott_2020_reformatted.csv"
)

sector = {
    "plane": input(
        "Local time sector (dawn, noon, dusk, midnight)?\n > "
    ),  # dawn, noon, dusk, midnight
}

# Start at first crossing
# Get the timedelta between the current crossing end and the next crossing start.
# Label the region by the crossing type of the next crossing.

times = []
region_labels = []

for i in range(len(crossings)):

    # Check if crossing is in sector, if not, continue
    match sector["plane"]:

        case "noon":
            if crossings.iloc[i]["start_x"] < abs(crossings.iloc[i]["start_y"]):
                continue

    try:
        time_to_next_crossing = (
            crossings.iloc[i + 1]["start"] - crossings.iloc[i]["end"]
        ).total_seconds() / 60  # minutes
    except:
        # If there are no more crossings, the next crossing doesn't exist. The search is complete!
        continue

    # We can check which region we are in by looking at the next crossing in the crossing list
    # In doing this, we assume Philpott+ (2020) didn't miss any crossings.
    # If they did, these would show up as outliers in the distributions
    match crossings.iloc[i + 1]["type"]:

        case "BS_IN":
            # We are in the solar wind
            region = "Solar Wind"

        case "MP_IN":
            # We are in the magnetosheath
            region = "Magnetosheath"

        case "MP_OUT":
            # We are in the magnetosphere
            region = "Magnetosphere"

        case "BS_OUT":
            # We are in the magnetosheath
            region = "Magnetosheath"

        case _:
            raise ValueError("Unknown crossing type!")

    times.append(time_to_next_crossing)
    region_labels.append(region)


# Convert to dataframe for ease of use
crossing_times = pd.DataFrame(
    {
        "label": region_labels,
        "length": times,
    }
)


# Setup figure
fig, axes = plt.subplots(1, 3, sharey=True, sharex=True)

region_options = list(set(region_labels))
region_options.sort()

bin_length = 15  # minutes

for i, ax in enumerate(axes):

    data = crossing_times[crossing_times["label"] == region_options[i]]["length"]

    ax.hist(
        data,
        bins=np.arange(np.min(data), np.max(data), bin_length),
        color="k",
        label=f"Number of passes = {len(data)}",
    )

    ax.set_title(region_options[i])
    ax.set_xlabel("Time To Cross Region [minutes]")

    ax.tick_params("x", which="major", direction="out", length=8, width=1)
    ax.tick_params("x", which="minor", direction="out", length=4, width=0.8)
    ax.tick_params("y", which="major", direction="out", length=8, width=1)
    ax.tick_params("y", which="minor", direction="out", length=4, width=0.8)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    ax.set_yscale("log")
    ax.set_xlim(0, 550)

    ax.margins(0)

    ax.legend()

    if i == 0:
        ax.set_ylabel("# Region Crossings")


plt.show()
