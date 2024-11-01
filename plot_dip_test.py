"""
Script to investigate the bimodality of a given bow shock crossing interval
"""

import diptest
import hermpy.boundary_crossings as boundaries
import hermpy.mag as mag
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# Load the crossing intervals
crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/Work/mercury/philpott_bow_shock_diptest.csv"
)

fig, axes = plt.subplots(1, 2, sharey=True)

for ax, label in zip(axes, ["dip_stat", "dip_p_value"]):
    ax.hist(crossings[label], color="black", bins=10, density=True)

    if ax == axes[0]:
        ax.set_xlabel("Hartigan's Dip Statistic")
        ax.set_ylabel("Percentage Crossings per Bin")

    else:
        ax.set_xlabel("Hartigan's Dip Statistic p-value")

    ax.margins(0)

plt.show()

"""
upper = 0.3
lower = 0.01
crossings = crossings.loc[ (crossings["dip_p_value"] > lower) & (crossings["dip_p_value"] < upper) ]

crossing = crossings.iloc[20]

data = mag.Load_Between_Dates("/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/", crossing["start"], crossing["end"], strip=True)

plt.plot(data["date"], data["mag_total"])
plt.show()
plt.hist(data["mag_total"])
plt.show()
"""

# MAKE SOME OTHER EXAMPLE DISTRIBUTIONS TO TAKE A DIP TEST OF

# Gaussian

gaussian = np.random.normal(size=10000)
bimodal = np.append(np.random.normal(loc = -3, size=5000), np.random.normal(loc = 3, size=5000))
trimodal = np.append( np.append( np.random.normal(loc = -5, size=3333), np.random.normal(loc = 5, size=3333) ), np.random.normal(size=3334) )
skewed = scipy.stats.skewnorm.rvs(20, size=10000)
uniform = np.random.uniform(size=10000)


fig, axes = plt.subplots(2, 2)

distributions = [gaussian, bimodal, skewed, uniform]
names = ["Gaussian", "Bimodal", "Skewed", "Uniform"]

for ax, distribution, name in zip(axes.flatten(), distributions, names):

    dip = diptest.diptest(distribution)
    label = f"{name}\n N = {len(distribution)}\n D = {dip[0]:.3f}\n p = {dip[1]:.3f}"

    ax.hist(distribution, label=label, bins=50, density=True)

    ax.legend()

plt.show()
