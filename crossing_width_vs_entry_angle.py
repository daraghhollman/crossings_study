"""
Script to plot the width of a BS crossing interval to the angle the entry trajectory makes with the surface normal
"""

import datetime as dt

import hermpy.trajectory as traj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from hermpy.utils import Constants
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def main():

    normalise_by_spacecraft_speed = True
    remove_outliers = True
    use_radii = False
    normalise_by_sim = False

    # Load the grazing angles
    print("Loading grazing angles...")
    crossings = pd.read_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/philpott_grazing_angles.csv",
        parse_dates=["Start Time", "End Time"],
    )

    crossings["Normal Vector"] = crossings["Normal Vector"].apply(
        lambda x: list(map(float, x.strip("[]").split()))
    )
    crossings["Spacecraft Velocity (x, rho)"] = crossings[
        "Spacecraft Velocity (x, rho)"
    ].apply(lambda x: list(map(float, x.strip("[]").split())))

    # Get the time difference between the start and the stop
    # crossings["dt"] = crossings.apply(Get_Crossing_Width, axis=1)
    print("Finding crossing interval lengths")
    crossings["dt"] = [
        t.total_seconds() / 60
        for t in (crossings["End Time"] - crossings["Start Time"]).tolist()
    ]  # minutes

    if normalise_by_spacecraft_speed:

        # Precompute positions:
        print("Precomputing positions")
        midpoint_times = (
            crossings["Start Time"]
            + (crossings["End Time"] - crossings["Start Time"]) / 2
        ).tolist()
        next_times = (
            crossings["Start Time"]
            + dt.timedelta(seconds=1)
            + (crossings["End Time"] - crossings["Start Time"]) / 2
        ).tolist()

        positions = traj.Get_Position(
            "MESSENGER", midpoint_times, frame="MSO", aberrate=False
        )
        next_positions = traj.Get_Position(
            "MESSENGER", next_times, frame="MSO", aberrate=False
        )

        print("Finding spacecraft velocity")
        velocities = next_positions - positions

        normal_vectors = np.vstack(crossings["Normal Vector"].tolist())
        cylindrical_velocities = np.vstack(
            crossings["Spacecraft Velocity (x, rho)"].tolist()
        )
        # print(normal_vectors[:,0])
        # print(cylindrical_velocities)

        speeds = np.sqrt(np.sum(velocities**2, axis=1))  # km/s

        velocity_normal_components = (
            np.abs(
                cylindrical_velocities[:, 0] * normal_vectors[:, 0]
                + cylindrical_velocities[:, 1] * normal_vectors[:, 1]
            )
            * speeds
        ) # Velocity normal to boundary, km/s

        print("Normalising crossing length by velocity")
        crossings["distance"] = (
            crossings["dt"] * 60 * velocity_normal_components
        )  # multiply minutes by 60 to get seconds

        # Set plotting params
        y_label = "Crossing Path Distance [km]"
        y_variable = "distance"
        degrees_bin_size = 2
        y_bin_size = 200  # km

        if use_radii:
            crossings["distance"] /= Constants.MERCURY_RADIUS_KM
            y_bin_size = 0.05  # R

    else:
        y_label = "Crossing Interval Length [minutes]"
        y_variable = "dt"
        degrees_bin_size = 2
        y_bin_size = 2  # minutes

    # remove outliers
    if remove_outliers:
        sigma = 5
        crossings = crossings.loc[
            (np.abs(scipy.stats.zscore(crossings[y_variable])) < sigma)
        ]

    mp_crossings = crossings.loc[crossings["Type"].str.contains("MP")]
    bs_crossings = crossings.loc[crossings["Type"].str.contains("BS")]

    print("Creating figure")
    fig, image_axes = plt.subplots(1, 2, sharey=True)

    bins = (
        np.arange(0, 90, degrees_bin_size),
        np.arange(0, crossings[y_variable].max(), y_bin_size),
    )

    for i, (image_axis, crossings) in enumerate(
        zip(image_axes, [bs_crossings, mp_crossings])
    ):

        heatmap, x_edges, y_edges = np.histogram2d(
            crossings["Grazing Angle (deg.)"], crossings[y_variable], bins=bins
        )

        """
        if normalise_by_sim:
            # Load sim data

            if i == 0:
                sim_grazing_angles = np.loadtxt(
                    "/home/daraghhollman/Main/Work/mercury/DataSets/expected_winslow_bow_shock_grazing_angles.txt"
                )

            else:
                sim_grazing_angles = np.loadtxt(
                    "/home/daraghhollman/Main/Work/mercury/DataSets/expected_winslow_magnetopause_grazing_angles.txt"
                )

            sim_histogram, _ = np.histogram(sim_grazing_angles, bins=bins[0])

            heatmap /= sim_histogram[:, None]
        """

        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        image = image_axis.imshow(
            heatmap.T, extent=extent, origin="lower", aspect="auto", norm="log"
        )

        ax1_divider = make_axes_locatable(image_axis)
        cax = ax1_divider.append_axes("right", size="5%", pad="2%")

        fig.colorbar(
            image,
            cax=cax,
            label=(
                "Num. Crossings"
                if not normalise_by_sim
                else "Num. Crossings, normalised by expected crossing grazing angles"
            ),
        )

        image_axis.set_ylabel(y_label)
        image_axis.set_xlabel("Grazing Angle [deg.]")

        image_axis.margins(0)

    plt.show()


def Get_Crossing_Width(row):
    return (row["End Time"] - row["Start Time"]).total_seconds() / 60  # minutes


if __name__ == "__main__":
    main()
