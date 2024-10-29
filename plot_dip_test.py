
"""
Script to investigate the bimodality of a given bow shock crossing interval
"""

import hermpy.boundary_crossings as boundaries
import hermpy.mag as mag
import matplotlib.pyplot as plt

# Load the crossing intervals
crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/Work/mercury/philpott_bow_shock_diptest.csv"
)

fig, axes = plt.subplots(1, 2, sharey=True)

for ax, label in zip(axes, ["dip_stat", "dip_p_value"]):
    ax.hist(crossings[label], color="black", bins = 10, density=True)


    if ax == axes[0]:
        ax.set_xlabel("Hartigan's Dip Statistic")
        ax.set_ylabel("Percentage Crossings per Bin")

    else:
        ax.set_xlabel("Hartigan's Dip Statistic p-value")


    ax.margins(0)

plt.show()

upper = 0.3
lower = 0.01
crossings = crossings.loc[ (crossings["dip_p_value"] > lower) & (crossings["dip_p_value"] < upper) ]

crossing = crossings.iloc[20]

data = mag.Load_Between_Dates("/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/", crossing["start"], crossing["end"], strip=True)

plt.plot(data["date"], data["mag_total"])
plt.show()
plt.hist(data["mag_total"])
plt.show()
