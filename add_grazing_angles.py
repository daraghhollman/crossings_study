"""
Script to determine and save grazing angles for each crossing in the Philpott crossing list
"""

import functools
import multiprocessing

import numpy as np
import pandas as pd
from hermpy import boundaries, trajectory, utils


def main():
    cores = 16
    aberrate = True

    # Load crossings
    crossings = boundaries.Load_Crossings(
        utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=False
    )

    bow_shock_crossings = crossings.loc[crossings["Type"].str.contains("BS")]
    magnetopause_crossings = crossings.loc[crossings["Type"].str.contains("MP")]

    # Get the grazing angle
    if aberrate:
        (
            bow_shock_crossings["Grazing Angle (deg.)"],
            bow_shock_crossings["Normal Vector"],
            bow_shock_crossings["Spacecraft Velocity (x, rho)"],
        ) = Parallelised_Apply(
            bow_shock_crossings, trajectory.Get_Bow_Shock_Grazing_Angle, cores
        )
        (
            magnetopause_crossings["Grazing Angle (deg.)"],
            magnetopause_crossings["Normal Vector"],
            magnetopause_crossings["Spacecraft Velocity (x, rho)"],
        ) = Parallelised_Apply(
            magnetopause_crossings, trajectory.Get_Magnetopause_Grazing_Angle, cores
        )

    elif aberrate == "average":
        (
            bow_shock_crossings["Grazing Angle (deg.)"],
            bow_shock_crossings["Normal Vector"],
            bow_shock_crossings["Spacecraft Velocity (x, rho)"],
        ) = trajectory.Get_Bow_Shock_Grazing_Angle(
            bow_shock_crossings, aberrate="average"
        )
        (
            magnetopause_crossings["Grazing Angle (deg.)"],
            magnetopause_crossings["Normal Vector"],
            magnetopause_crossings["Spacecraft Velocity (x, rho)"],
        ) = trajectory.Get_Magnetopause_Grazing_Angle(
            magnetopause_crossings, aberrate="average"
        )

    # Create subset dataframe
    bow_shock_crossings = bow_shock_crossings[
        [
            "Type",
            "Start Time",
            "End Time",
            "Grazing Angle (deg.)",
            "Normal Vector",
            "Spacecraft Velocity (x, rho)",
        ]
    ]
    magnetopause_crossings = magnetopause_crossings[
        [
            "Type",
            "Start Time",
            "End Time",
            "Grazing Angle (deg.)",
            "Normal Vector",
            "Spacecraft Velocity (x, rho)",
        ]
    ]

    # Combine together and sort
    combined_crossings = pd.concat([bow_shock_crossings, magnetopause_crossings])
    combined_crossings = combined_crossings.sort_values("Start Time")

    # Save to csv
    if aberrate == "average":
        combined_crossings.to_csv(
            "/home/daraghhollman/Main/Work/mercury/DataSets/philpott_grazing_angles_average_aberration.csv"
        )

    else:
        combined_crossings.to_csv(
            "/home/daraghhollman/Main/Work/mercury/DataSets/philpott_grazing_angles.csv"
        )


# Functions to use pandas' apply with parallelisation
def Parallelise(data, function, number_of_processes):
    data_splits = np.array_split(data, number_of_processes)

    with multiprocessing.Pool(number_of_processes) as pool:
        data = pd.concat(pool.map(function, data_splits))

    try:
        data = data.iloc[:, 0].to_numpy()

    except pd.errors.IndexingError:
        print(data)

    output = [[], [], []]
    for row in data:
        for i in range(len(output)):
            output[i].append(row[i])

    return output


def Apply_Function_To_Subset(function, data_subset):
    return data_subset.apply(function, axis=1)


def Parallelised_Apply(data, function, number_of_processes):
    return Parallelise(
        data, functools.partial(Apply_Function_To_Subset, function), number_of_processes
    )


if __name__ == "__main__":
    main()
