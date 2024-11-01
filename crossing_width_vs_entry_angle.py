"""
Script to plot the width of a BS crossing interval to the angle the entry trajectory makes with the surface normal
"""

import datetime as dt

import hermpy.boundary_crossings as boundaries
import hermpy.trajectory as traj
import hermpy.mag as mag
import hermpy.plotting_tools as hermplot
import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import scipy.stats

spice.furnsh("/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt")


# Load the crossing intervals
crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/Work/mercury/philpott_2020_reformatted.csv"
)

# Limit only to bow shock crossings
crossings = crossings.loc[
    (crossings["type"] == "BS_OUT") | (crossings["type"] == "BS_IN")
]


# Get the time difference between the start and the stop
def Get_Crossing_Width(row):
    return (row["end"] - row["start"]).total_seconds() / 60 # minutes


crossings["dt"] = crossings.apply(Get_Crossing_Width, axis=1)


# Get the angle between the trajectory and the bs normal
def Get_Surface_Angle(row):

    # Get position at start of crossing
    start_position = traj.Get_Position("MESSENGER", row["start"], frame="MSM")
    next_position = traj.Get_Position(
        "MESSENGER", row["start"] + dt.timedelta(seconds=1), frame="MSM"
    )

    velocity = next_position - start_position

    # We find the closest position on the Winslow (2013) BS model
    initial_x = 0.5
    psi = 1.04
    p = 2.75

    L = psi * p

    phi = np.linspace(0, 2 * np.pi, 1000)
    rho = L / (1 + psi * np.cos(phi))

    bow_shock_x_coords = initial_x + rho * np.cos(phi)
    bow_shock_y_coords = rho * np.sin(phi)
    bow_shock_z_coords = rho * np.sin(phi)

    bow_shock_positions = np.array(
        [bow_shock_x_coords, bow_shock_y_coords, bow_shock_z_coords]
    ).T

    # Initialise some crazy big value
    shortest_distance = 10000
    closest_position = 0

    for i, bow_shock_position in enumerate(bow_shock_positions):

        distance_to_position = np.sqrt(
            (start_position[0] - bow_shock_position[0]) ** 2
            + (start_position[1] - bow_shock_position[1]) ** 2
            + (start_position[2] - bow_shock_position[2]) ** 2
        )

        if distance_to_position < shortest_distance:
            shortest_distance = distance_to_position
            closest_position = i

        else:
            continue

    # Get the normal vector of the BS at this point
    # This is just the normalised vector between the spacecraft and the closest point
    normal_vector = bow_shock_positions[closest_position] - start_position
    normal_vector = normal_vector / np.sqrt(np.sum(normal_vector**2))

    grazing_angle = np.arccos(
        np.dot(normal_vector, velocity)
        / (np.sqrt(np.sum(normal_vector**2)) + np.sqrt(np.sum(velocity**2)))
    ) * 180 / np.pi

    # If the grazing angle is greater than 90, then we take 180 - angle as its from the other side
    # This occurs as we don't make an assumption as to what side of the model boundary we are

    if grazing_angle > 90:
        grazing_angle = 180 - grazing_angle

    return grazing_angle


crossings["grazing angle"] = crossings.apply(Get_Surface_Angle, axis=1)

# Remove outliers
crossings = crossings[np.abs(scipy.stats.zscore(crossings["dt"])) < 3]

correlation = scipy.stats.pearsonr(crossings["grazing angle"], crossings["dt"])



fig, image_axis = plt.subplots()


heatmap, x_edges, y_edges = np.histogram2d(crossings["grazing angle"], crossings["dt"], bins=30)
extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

image = image_axis.imshow(heatmap.T, extent=extent, origin="lower", norm="log", aspect="auto")

ax1_divider = make_axes_locatable(image_axis)
cax = ax1_divider.append_axes("right", size="5%", pad="2%")

fig.colorbar(image, cax=cax, label="Num. Crossings")

image_axis.set_ylabel("Crossing Interval Length [minutes]")
image_axis.set_xlabel("Grazing Angle [deg.]")

image_axis.margins(0)


plt.show()


"""
fig, (image_axis, line_axis) = plt.subplots(2, 1, height_ratios=[3, 1], sharex=True)


heatmap, x_edges, y_edges = np.histogram2d(crossings["grazing angle"], crossings["dt"], bins=20)
extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

image = image_axis.imshow(heatmap.T, extent=extent, origin="lower", norm="log", aspect="auto")

ax1_divider = make_axes_locatable(image_axis)
cax = ax1_divider.append_axes("right", size="5%", pad="2%")
ax2_divider = make_axes_locatable(line_axis)
_ = ax2_divider.append_axes("right", size="5%", pad="2%")
_.axis("off")

fig.colorbar(image, cax=cax, label="Num. Crossings")

image_axis.set_ylabel("Crossing Interval Length [minutes]")
line_axis.set_ylabel("Column Standard Deviation")
line_axis.set_xlabel("Grazing Angle [deg.]")

image_axis.margins(0)
line_axis.margins(0)


### Standard deviation of columns as a function of grazing angle
standard_deviations = scipy.stats.std(heatmap, axis=0)

x_centres = (x_edges[:-1] + x_edges[1:]) / 2
line_axis.plot(x_centres, standard_deviations)


plt.show()
"""
