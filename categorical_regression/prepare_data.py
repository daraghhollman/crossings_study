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
import pandas as pd
import numpy as np
import spiceypy as spice

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
data = mag.Strip_Data(data, mission_start, mission_end)
# data = mag.Add_Field_Variability(data, dt.timedelta(seconds=20), multiprocess=False)

region_data = pd.DataFrame(columns=["region", "mag_total", "mag_x", "mag_y", "mag_z", "mag_variability"])

# Check if we are sufficiently close to the bondaries
boundary_distance = dt.timedelta(minutes=5)


def Add_Data_Row(row):
    # We must first check which region we are in:
    # we can do this by finding the next crossing in the Philpott list
    try:
        next_crossing = philpott_crossings.loc[
            philpott_crossings["start"].between(row["date"], mission_end)
        ].iloc[0]
    except:
        return
    try:
        previous_crossing = philpott_crossings.loc[
            philpott_crossings["start"].between(mission_start, row["date"])
        ].iloc[-1]
    except:
        return
    
    if not (next_crossing["start"] - row["date"]) < boundary_distance or (row["date"] - previous_crossing["end"]) < boundary_distance:
        return

    # Check if we are within an interval
    row_inside_interval = philpott_crossings[
        (philpott_crossings["start"] < row["date"])
        & (philpott_crossings["end"] > row["date"])
    ]
    if len(row_inside_interval) > 0:
        # We are inside a crossing interval and need to skip this row
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
    region_dict = [
        region,
        row["mag_total"],
        row["mag_x"],
        row["mag_y"],
        row["mag_z"],
        row["mag_variability"]
    ]
    return region_dict


print("Iterrating through data")

count = 0
items = [row for _, row in data.iterrows()]
with multiprocessing.Pool() as pool:
    for result in pool.imap(Add_Data_Row, items):
        if result != None:
            region_data.loc[len(region_data.index)] = result
        count += 1
        print(f"{count} / {len(data)}", end="\r")


# Save to csv
region_data.to_csv(f"./region_data_{int(len(data) / 3600 / 24)}_{int(boundary_distance.total_seconds() / 60)}_mins.csv")
