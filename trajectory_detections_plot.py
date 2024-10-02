"""
Creates a plot in the xy and xz planes of the entire orbital trajectory and all labelled crossings
"""

import datetime as dt

import hermpy.boundary_crossings as boundaries
import hermpy.plotting_tools as plotting
import hermpy.trajectory as trajectory
import matplotlib as mpl
import matplotlib.pyplot as plt
import spiceypy as spice
from tqdm import tqdm

from plotting_tools import colored_line

mpl.rcParams["font.size"] = 14
root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"
metakernel = "/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt"
spice.furnsh(metakernel)
philpott_crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/mercury/philpott_2020_reformatted.csv"
)

start_time = dt.datetime(year=2011, month=3, day=23, hour=16)
end_time = dt.datetime(year=2015, month=4, day=30, hour=15)

# Get trajectory for entire mission in minutes
positions = trajectory.Get_Trajectory(
    "MESSENGER",
    [start_time, end_time],
    steps=int((end_time - start_time).total_seconds() / 1800),
    frame="MSM",
    aberrate=True,
)

# Convert from km to Mercury radii
positions /= 2439.7

# Plot positions
fig, axes = plt.subplots(1, 2)

xy_axis, xz_axis = axes

# To get nice overlap rendering, we can break the curve up into it's individual line segments
for i in tqdm(range(len(positions) - 1), total=len(positions) - 1):
    xy_axis.plot(
        positions[i : i + 2][:, 0],
        positions[i : i + 2][:, 1],
        color="k",
        alpha=0.05,
        solid_capstyle="butt",
        zorder=-1,
    )

    # Would you believe, 70k axis labels is a lot to fit in a legend
    if i == 0:
        xz_axis.plot(
            positions[i : i + 2][:, 0],
            positions[i : i + 2][:, 2],
            color="k",
            alpha=0.05,
            solid_capstyle="butt",
            zorder=-1,
            label="MESSENGER Trajectory",
        )
    else:
        xz_axis.plot(
            positions[i : i + 2][:, 0],
            positions[i : i + 2][:, 2],
            color="k",
            alpha=0.05,
            solid_capstyle="butt",
            zorder=-1,
        )


planes = ["xy", "xz"]
for i, ax in enumerate(axes):
    plotting.Plot_Mercury(
        ax,
        shaded_hemisphere="left",
        plane=planes[i],
        frame="MSM",
        border_colour="grey",
    )
    plotting.Add_Labels(ax, planes[i], frame="MSM", aberrate=True)
    plotting.Plot_Magnetospheric_Boundaries(ax, plane=planes[i], frame="MSM", add_legend=True, zorder=15)
    plotting.Square_Axes(ax, 8)


"""
# We then want to get the boundaries from Philpott and plot those positions
bow_shock_data = philpott_crossings[(philpott_crossings["type"] == "BS_IN") | (philpott_crossings["type"] == "BS_OUT")]
magnetopause_data = philpott_crossings[(philpott_crossings["type"] == "MP_IN") | (philpott_crossings["type"] == "MP_OUT")]

xy_axis.plot(
    bow_shock_data["start_x"] / 2439.7,
    bow_shock_data["start_y"] / 2439.7,
    "+",
    color="mediumturquoise",
    label="Philpott Bowshock Crossings",
    alpha=0.5,
    zorder=10
)
xy_axis.plot(
    magnetopause_data["start_x"] / 2439.7,
    magnetopause_data["start_y"] / 2439.7,
    "+",
    color="indianred",
    label="Philpott Magnetopause Crossings",
    alpha=0.5,
    zorder=10
)
xz_axis.plot(
    bow_shock_data["start_x"] / 2439.7,
    bow_shock_data["start_z"] / 2439.7,
    "+",
    color="mediumturquoise",
    label="Philpott Bowshock Crossings",
    alpha=0.5,
    zorder=10
)
xz_axis.plot(
    magnetopause_data["start_x"] / 2439.7,
    magnetopause_data["start_z"] / 2439.7,
    "+",
    color="indianred",
    label="Philpott Magnetopause Crossings",
    alpha=0.5,
    zorder=10
)
"""

xz_axis.legend(bbox_to_anchor=(-0.1, 1.1), loc="center", ncol=2, borderaxespad=0.5)

plt.show()
