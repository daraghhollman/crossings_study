"""
Script to load the philpott boundaries list, and load and save the solar wind time series data in a fixed length window before/after the boundary, along with any other raw data such as ephemeris, local time, latitude, magnetic latitude
"""

import datetime as dt
import multiprocessing

import numpy as np
import pandas as pd
import scipy.stats


# Load samples csv 
samples_data_set = pd.read_csv("./solar_wind_sample_10_mins.csv")

# Filter to magnetic north
samples_data_set = samples_data_set.loc[ samples_data_set["magnetic_latitude"] > 0 ]

# Filter by local time to between 11 and 13 LT
samples_data_set = samples_data_set.loc[ (samples_data_set["local_time"] >= 11) & (samples_data_set["local_time"] <= 13) ]


# Fix loading issues to do with an element being a series itself
samples_data_set["mag_total"] = samples_data_set["mag_total"].apply(lambda x: list(map(float, x.strip("[]").split(','))))
samples_data_set["mag_x"] = samples_data_set["mag_x"].apply(lambda x: list(map(float, x.strip("[]").split(','))))
samples_data_set["mag_y"] = samples_data_set["mag_y"].apply(lambda x: list(map(float, x.strip("[]").split(','))))
samples_data_set["mag_z"] = samples_data_set["mag_z"].apply(lambda x: list(map(float, x.strip("[]").split(','))))


def Get_Features(row):
    """
    For a paricular solar wind sample (or 'event'),
    return a dictionary of all the relevant raw properties.
    """

    # Each feature return will be a tuple with the calculation for each value
    # Find the mean
    mean = np.mean([row["mag_total"], row["mag_x"], row["mag_y"], row["mag_z"]], axis=1)
    median = np.median([row["mag_total"], row["mag_x"], row["mag_y"], row["mag_z"]], axis=1)
    std = np.std([row["mag_total"], row["mag_x"], row["mag_y"], row["mag_z"]], axis=1)
    skew = scipy.stats.skew([row["mag_total"], row["mag_x"], row["mag_y"], row["mag_z"]], axis=1)
    kurtosis = scipy.stats.kurtosis([row["mag_total"], row["mag_x"], row["mag_y"], row["mag_z"]], axis=1)

    return {
        # Time identifiers
        "crossing_start": row["crossing_start"],
        "crossing_end": row["crossing_end"],
        "sample_start": row["sample_start"],
        "sample_end": row["sample_end"],

        # Parameters
        "mean": mean,
        "median": median,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
    }


# Create empty list of dictionaries
solar_wind_features = []

# Iterrate through the crossings
count = 0
process_items = [row for _, row in samples_data_set.iterrows()]
with multiprocessing.Pool() as pool:
    for result in pool.imap(Get_Features, process_items):


        if result is not None:
            # Add row dictionary to list
            solar_wind_features.append(result)


        count += 1
        print(f"{count} / {len(samples_data_set)}", end="\r")


# Create dataframe from solar wind samples
solar_wind_features = pd.DataFrame(solar_wind_features)

print("")

solar_wind_features.to_csv(f"./noon_north_filtered_solar_wind_features.csv")
