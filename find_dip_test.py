
"""
Script to investigate the bimodality of a given bow shock crossing interval
"""

import hermpy.boundary_crossings as boundaries
import hermpy.mag as mag
import diptest
import multiprocessing

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

    if len(data["mag_total"]) < 4:
        return None, None

    # Test for unimodality in |B|:
    dip, p = diptest.diptest(data["mag_total"])

    return dip, p

# Iterrate through the crossings
dips = []
p_values = []
count = 0
process_items = [row for _, row in crossings.iterrows()]
with multiprocessing.Pool(int(input("# of cores? "))) as pool:
    for dip, p in pool.imap(Get_Dip_Test, process_items):

        dips.append(dip)
        p_values.append(p)

        count += 1
        print(f"{count} / {len(crossings)}", end="\r")


crossings["dip_stat"] = dips
crossings["dip_p_value"] = p_values

crossings.to_csv("/home/daraghhollman/Main/Work/mercury/philpott_bow_shock_diptest.csv")
