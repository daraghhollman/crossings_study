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

    # Load crossings
    crossings = boundaries.Load_Crossings(
        User.CROSSING_LISTS["Philpott"], include_data_gaps=False
    )

    # Get the grazing angle
    crossings["Grazing Angle (deg.)"] = Parallelised_Apply(
        crossings, traj.Get_Grazing_Angle, int(input("# of cores? "))
    )

    # Create subset dataframe
    crossings = crossings[["Type", "Start Time", "End Time", "Grazing Angle (deg.)"]]

    # Save to csv
    crossings.to_csv(
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
