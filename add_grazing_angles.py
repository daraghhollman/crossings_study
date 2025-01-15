"""
Script to determine and save grazing angles for each crossing in the Philpott crossing list
"""

import functools
import multiprocessing

import hermpy.boundary_crossings as boundaries
import hermpy.trajectory as traj
import numpy as np
import pandas as pd
from hermpy.utils import User


def main():
    cores = int(input("# of cores? "))
    aberrate = True

    # Load crossings
    crossings = boundaries.Load_Crossings(
        User.CROSSING_LISTS["Philpott"], include_data_gaps=False
    )

    bow_shock_crossings = crossings.loc[ crossings["Type"].str.contains("BS") ]
    magnetopause_crossings = crossings.loc[ crossings["Type"].str.contains("MP") ]


    # Get the grazing angle
    if aberrate:
        bow_shock_crossings["Grazing Angle (deg.)"] = Parallelised_Apply(
            bow_shock_crossings, traj.Get_Bow_Shock_Grazing_Angle, cores
        )
        magnetopause_crossings["Grazing Angle (deg.)"] = Parallelised_Apply(
            magnetopause_crossings, traj.Get_Magnetopause_Grazing_Angle, cores
        )

    elif aberrate == "average":
        bow_shock_crossings["Grazing Angle (deg.)"] = traj.Get_Bow_Shock_Grazing_Angle(bow_shock_crossings, aberrate="average")
        magnetopause_crossings["Grazing Angle (deg.)"] = traj.Get_Magnetopause_Grazing_Angle(magnetopause_crossings, aberrate="average")


    # Create subset dataframe
    bow_shock_crossings = bow_shock_crossings[["Type", "Start Time", "End Time", "Grazing Angle (deg.)"]]
    magnetopause_crossings = magnetopause_crossings[["Type", "Start Time", "End Time", "Grazing Angle (deg.)"]]

    # Combine together and sort
    combined_crossings = pd.concat([bow_shock_crossings, magnetopause_crossings])
    combined_crossings.sort_values("Start Time")

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

    return data


def Apply_Function_To_Subset(function, data_subset):
    return data_subset.apply(function, axis=1)


def Parallelised_Apply(data, function, number_of_processes):
    return Parallelise(
        data, functools.partial(Apply_Function_To_Subset, function), number_of_processes
    )


if __name__ == "__main__":
    main()
