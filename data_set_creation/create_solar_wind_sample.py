"""
Script to load the philpott boundaries list, and load and save the solar wind time series data in a fixed length window before/after the boundary, along with any other raw data such as ephemeris, local time, latitude, magnetic latitude
"""

import datetime as dt
import multiprocessing

import numpy as np
import pandas as pd
import hermpy.boundary_crossings as boundaries
import hermpy.mag as mag
import spiceypy as spice

spice.furnsh("/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt")


root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"
sample_length = dt.timedelta(minutes=10)


# Load Philpott+ (2020) crossings
crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/Work/mercury/philpott_2020_reformatted.csv"
)

crossings = crossings.loc[( crossings["type"] == "BS_OUT" ) | ( crossings["type"] == "BS_IN" )]

def Get_Sample(row):
    """
    For a paricular solar wind sample (or 'event'),
    return a dictionary of all the relevant raw properties.
    """

    # Find out if solar wind is before or after based on the type of the current crossing
    if row["type"] == "BS_OUT":
        # Solar wind is after the crossing
        sample_start = row["end"]
        sample_end = row["end"] + sample_length

    elif row["type"] == "BS_IN":
        # Solar wind is before the crossing
        sample_start = row["start"] - sample_length
        sample_end = row["start"]

    else:
        return None




    # Load sample data:
    sample = mag.Load_Between_Dates(root_dir, sample_start, sample_end, strip=True)

    sample_position = [sample["eph_x"], sample["eph_y"], sample["eph_z"]]
    sample_middle_position = sample_position[len(sample_position) // 2]

    # convert to radii from km
    sample_middle_position /= 2439.7

    longitude = np.arctan2(sample_middle_position[1], sample_middle_position[0]) * 180 / np.pi

    if longitude < 0:
        longitude += 360

    local_time = ((longitude + 180) * 24 / 360) % 24

    latitude = np.arctan2(
        sample_middle_position[2], np.sqrt(sample_middle_position[0] ** 2 + sample_middle_position[1] ** 2)
    ) * 180 / np.pi

    magnetic_latitude = np.arctan2(
        sample_middle_position[2] - (479 / 2439.7), np.sqrt(sample_middle_position[0] ** 2 + sample_middle_position[1] ** 2)
    ) * 180 / np.pi


    return {
        # Time identifiers
        "crossing_start": row["start"],
        "crossing_end": row["end"],
        "sample_start": sample_start,
        "sample_end": sample_end,

        # Data sample itself
        "UTC": sample["date"].tolist(),
        "|B|": sample["mag_total"].tolist(),
        "B_x": sample["mag_x"].tolist(),
        "B_y": sample["mag_y"].tolist(),
        "B_z": sample["mag_z"].tolist(),

        # The mean local time of the sample
        "LT": local_time,

        # The mean latitude of the sample
        "Lat": latitude,

        # The mean magnetic latitude of the sample
        "MLat": magnetic_latitude,
    }


# Create empty list of dictionaries
solar_wind_samples = []

# Iterrate through the crossings
count = 0
process_items = [row for _, row in crossings.iterrows()]
with multiprocessing.Pool() as pool:
    for result in pool.imap(Get_Sample, process_items):


        if result is not None:
            # Add row dictionary to list
            solar_wind_samples.append(result)


        count += 1
        print(f"{count} / {len(crossings)}", end="\r")


# Create dataframe from solar wind samples
solar_wind_samples = pd.DataFrame(solar_wind_samples)

print("")

solar_wind_samples.to_csv(f"./solar_wind_sample_{int(sample_length.total_seconds() / 60)}_mins.csv")
