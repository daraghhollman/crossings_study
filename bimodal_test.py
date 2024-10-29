
"""
Script to investigate the bimodality of a given bow shock crossing interval
"""

import datetime as dt

import hermpy.boundary_crossings as boundaries
import hermpy.trajectory as traj
import hermpy.mag as mag
import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import scipy.stats

import diptest
import multiprocessing
import pandas as pd

spice.furnsh("/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt")

root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"

# Load the crossing intervals
crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/Work/mercury/philpott_2020_reformatted.csv"
)

# Limit only to bow shock crossings
crossings = crossings.loc[
    (crossings["type"] == "BS_OUT") | (crossings["type"] == "BS_IN")
]


def Get_Dip_Test(row):
    # Load the data within the distribution
    data = mag.Load_Between_Dates(root_dir, row["start"], row["end"], strip=True)

    # Test for unimodality in |B|:
    dip = diptest.dipstat(data["mag_total"])

    return dip

def Apply_Process(df):
    result = df.apply(Get_Dip_Test, axis=1)
    return result


pool = multiprocessing.Pool(processes=20)
split_crossings = np.array_split(crossings, 20)
pool_results = pool.map(Apply_Process, split_crossings)
pool.close()
pool.join()

# merging parts processed by different processes
parts = pd.concat(pool_results, axis=0)

# merging newly calculated parts to big_df
crossings["diptest"] = parts

# crossings["diptest"] = crossings.apply(Get_Dip_Test, axis=1)


fig, ax = plt.subplots()

ax.hist(crossings["diptest"], color="black")

ax.set_xlabel("Hartigan's Dip Statistic")
ax.set_ylabel("Number of Crossings")

ax.margins(0)

plt.show()
