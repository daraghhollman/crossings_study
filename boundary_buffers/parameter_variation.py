"""
Script to plot how different parameters of the solar wind distribution change as a function of boundary buffer
"""

import datetime as dt
import multiprocessing

import hermpy.boundary_crossings as boundaries
import hermpy.mag as mag
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["font.size"] = 12
root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"

sector = {
    "local_time": "noon",  # dawn, noon, dusk, midnight
    "hemisphere": "north",  # north, south
    "boundary": "bs_out",
}

crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/Work/mercury/philpott_2020_reformatted.csv"
)

# Filter crossings list by hemisphere
match sector["hemisphere"]:

    case "north":
        crossings = crossings.loc[crossings["start_z"] > 479]  # Offset for MSM

    case "south":
        crossings = crossings.loc[crossings["start_z"] < 479]  # Offset for MSM

# Filter crossings list by local time
match sector["local_time"]:

    case "dawn":
        crossings = crossings.loc[crossings["start_y"] < -abs(crossings["start_x"])]

    case "noon":
        # Noon is between the lines x = y and x = -y
        # i.e. regions where x > |y|
        crossings = crossings.loc[crossings["start_x"] > abs(crossings["start_y"])]

    case "dusk":
        crossings = crossings.loc[crossings["start_y"] > abs(crossings["start_x"])]

    case "midnight":
        crossings = crossings.loc[crossings["start_x"] < -abs(crossings["start_y"])]

match sector["boundary"]:

    case "bs":
        crossings = crossings.loc[
            (crossings["type"] == "BS_OUT") | (crossings["type"] == "BS_IN")
        ]

    case "bs_out":
        crossings = crossings.loc[(crossings["type"] == "BS_OUT")]

    case "bs_in":
        crossings = crossings.loc[(crossings["type"] == "BS_IN")]

    case "mp":
        crossings = crossings.loc[
            (crossings["type"] == "MP_OUT") | crossings["type"] == "MP_IN"
        ]


# We iterrate through the crossings in the shortened dataframe.
# We determine the distribution of the mag_total data just outside the boundary interval,
# and at increasing time buffers.
# We then determine some parameters for these distributions and plot them.

buffers = [dt.timedelta(minutes=int(i)) for i in np.arange(0, 20, 1)]  # minutes
sample_length = dt.timedelta(minutes=10)  # minutes

# shorten crossings to test with
# crossings = crossings.iloc[0 : int(len(crossings) / 40)]

overall_means = []
overall_medians = []
overall_stds = []
overall_fwhms = []


def Get_Parameters(row):
    # Load the data for the day of this crossing
    # We add the largest buffered window to the end in case we go past the day
    start_time = row["start"]
    end_time = row["end"] + buffers[-1] + sample_length

    data = mag.Load_Between_Dates(
        root_dir, start_time, end_time, strip=True, verbose=False
    )

    sample_means = []
    sample_medians = []
    sample_stds = []

    for buffer in buffers:
        # Get data within window
        distribution_data = data.loc[
            data["date"].between(
                row["end"] + buffer, row["end"] + buffer + sample_length
            )
        ]["mag_total"]

        sample_means.append(np.mean(distribution_data))
        sample_medians.append(np.median(distribution_data))
        sample_stds.append(np.std(distribution_data))

    return sample_means, sample_medians, sample_stds


count = 0
process_items = [row for _, row in crossings.iterrows()]
with multiprocessing.Pool() as pool:
    for result in pool.imap(Get_Parameters, process_items):

        overall_means.append(result[0])
        overall_medians.append(result[1])
        overall_stds.append(result[2])

        count += 1
        print(f"{count} / {len(crossings)}", end="\r")


titles = ["mean [nT]", "median [nT]", "std [nT]"]
items_to_plot = [overall_means, overall_medians, overall_stds]
fig, axes = plt.subplots(1, len(titles))

for i, (ax, items) in enumerate(zip(axes, items_to_plot)):
    for j in range(len(items)):

        ax.plot(
            [k.total_seconds() / 60 for k in buffers],
            items[j],
            color="black",
            alpha=0.2,
        )

    ax.plot(
        [k.total_seconds() / 60 for k in buffers],
        np.median(items, axis=0),
        color="magenta",
        label="median",
    )
    ax.plot(
        [k.total_seconds() / 60 for k in buffers],
        np.mean(items, axis=0),
        color="magenta",
        ls="dashed",
        label="mean",
    )

    ax.annotate(
        f"N={len(items)}",
        xy=(1, 1),
        xycoords="axes fraction",
        size=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", fc="w"),
    )

    ax.margins(0)
    ax.set_xlabel("Boundary Interval Buffer [mins]")
    ax.set_title(titles[i])
    ax.legend()

    if i == 0:
        ax.set_ylabel("|B| [nT]")


plt.show()
