"""
Before starting a categorical regression model, we want to prepare our data.

Possible components to include:
    |B|
    Bx
    By
    Bz
    B_variability
    dBx/dBy
    dBx/dBz

We want to search outside of the Philpott boundary intervals, and add the data to corresponding dataframes
"""

import datetime as dt
import multiprocessing

import hermpy.boundary_crossings as boundaries
import hermpy.mag as mag
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import spiceypy as spice
from tqdm import tqdm

mpl.rcParams["font.size"] = 15
root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"
metakernel = "/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt"
spice.furnsh(metakernel)
philpott_crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/mercury/philpott_2020_reformatted.csv"
)

mission_start = dt.datetime(year=2012, month=6, day=1)
mission_end = dt.datetime(year=2012, month=6, day=2)

# 1: LOAD DATA
# Here it would take too long to adjust for aberration
data = mag.Load_Between_Dates(root_dir, mission_start, mission_end)

region_data = pd.DataFrame(columns=["region", "mag_total", "mag_x", "mag_y", "mag_z"])


def Add_Data_Row(row):
    # Check if we are within an interval
    row_inside_interval = philpott_crossings[
        (philpott_crossings["start"] < row["date"])
        & (philpott_crossings["end"] > row["date"])
    ]
    if len(row_inside_interval) > 0:
        # We are inside a crossing interval and need to skip this row
        return

    # We must first check which region we are in:
    # we can do this by finding the next crossing in the Philpott list
    try:
        next_crossing = philpott_crossings.loc[
            philpott_crossings["start"].between(row["date"], mission_end)
        ].iloc[0]
    except:
        return

    match next_crossing["type"]:

        case "BS_IN":
            # We are in the solar wind
            region = "solar_wind"

        case "MP_IN":
            # We are in the magnetosheath
            region = "magnetosheath"

        case "MP_OUT":
            # We are in the magnetosphere
            region = "magnetosphere"

        case "BS_OUT":
            # We are in the magnetosheath
            region = "magnetosheath"

        case _:
            raise ValueError("Unknown crossing type!")

    # Now we add all of the parameters we want to a new dataframe
    region_data.loc[len(region_data.index)] = [
        region,
        row["mag_total"],
        row["mag_x"],
        row["mag_y"],
        row["mag_z"],
    ]


print("Iterrating through data")

count = 0
items = [row for _, row in data.iterrows()]
with multiprocessing.Pool() as pool:
    for result in pool.imap(Add_Data_Row, items):
        count += 1
        print(f"{count} / {len(data)}", end="\r")


# Save to csv
region_data.to_csv("./region_data.csv")
