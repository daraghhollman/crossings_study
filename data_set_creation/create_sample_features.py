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

# Fix loading issues to do with an element being a series itself
samples_data_set["|B|"] = samples_data_set["|B|"].apply(lambda x: list(map(float, x.strip("[]").split(','))))
samples_data_set["B_x"] = samples_data_set["B_x"].apply(lambda x: list(map(float, x.strip("[]").split(','))))
samples_data_set["B_y"] = samples_data_set["B_y"].apply(lambda x: list(map(float, x.strip("[]").split(','))))
samples_data_set["B_z"] = samples_data_set["B_z"].apply(lambda x: list(map(float, x.strip("[]").split(','))))

def Get_Features(row):
    """
    For a paricular solar wind sample (or 'event'),
    return a dictionary of all the relevant raw properties.
    """

    # Each feature return will be a list with the calculation for each component
    mean = np.mean([row["|B|"], row["B_x"], row["B_y"], row["B_z"]], axis=1)
    median = np.median([row["|B|"], row["B_x"], row["B_y"], row["B_z"]], axis=1)
    std = np.std([row["|B|"], row["B_x"], row["B_y"], row["B_z"]], axis=1)
    skew = scipy.stats.skew([row["|B|"], row["B_x"], row["B_y"], row["B_z"]], axis=1)
    kurtosis = scipy.stats.kurtosis([row["|B|"], row["B_x"], row["B_y"], row["B_z"]], axis=1)

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
        "LT": row["LT"],
        "Lat": row["Lat"],
        "MLat": row["MLat"],
        "x_msm": row["x_msm"],
        "y_msm": row["y_msm"],
        "z_msm": row["z_msm"],
    }


# Create empty list of dictionaries
features_data = []

# Iterrate through the crossings
count = 0
process_items = [row for _, row in samples_data_set.iterrows()]
with multiprocessing.Pool() as pool:
    for result in pool.imap(Get_Features, process_items):


        if result is not None:
            # Add row dictionary to list
            features_data.append(result)


        count += 1
        print(f"{count} / {len(samples_data_set)}", end="\r")


# Create dataframe from solar wind samples
features_data_set = pd.DataFrame(features_data)

print("")

features_data_set.to_csv(f"./solar_wind_features.csv")
