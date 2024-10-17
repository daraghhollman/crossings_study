"""
Script to load the philpott boundaries list, and load and save the solar wind time series data in a fixed length window before/after the boundary, along with any other raw data such as ephemeris, local time, latitude, magnetic latitude
"""

import datetime as dt
import multiprocessing

import numpy as np
import pandas as pd
import hermpy.boundary_crossings as boundaries
import hermpy.mag as mag
import hermpy.trajectory as trajectory
import spiceypy as spice

spice.furnsh("/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt")


root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"
sample_length = dt.timedelta(minutes=10)


# Load Philpott+ (2020) crossings
crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/Work/mercury/philpott_2020_reformatted.csv"
)
crossings = crossings.loc[crossings["start_z"] > 479] # Offset for MSM
one_hour_factor = np.tan(15 * np.pi / 180)
crossings = crossings.loc[crossings["start_x"] > (1 / one_hour_factor) * abs(crossings["start_y"])]

# Limit by year
crossings = crossings.loc[ crossings["start"].dt.year == dt.date(year=2014, month=1, day=1).year ]

# Limit by heliocentric distance
heliocentric_distances = []
for i, row in crossings.iterrows():

    r = trajectory.Get_Heliocentric_Distance(row["start"])

    heliocentric_distances.append(r / 1.496e+8)

crossings["heliocentric_distance"] = heliocentric_distances

crossings = crossings.loc[ (crossings["heliocentric_distance"] > 0.35) & (crossings["heliocentric_distance"] < 0.4) ]

def Get_Sample(row):
    """
    For a paricular solar wind sample (or 'event'),
    return a dictionary of all the relevant raw properties.
    """

    current_index = crossings.index.get_loc(row.name)

    # Find out if magnetosheath is before or after based on the type of the current crossing
    if row["type"] == "BS_IN":
        sample_start = row["end"]
        sample_end = row["end"] + sample_length

        # we need to enure we don't go past into the magnetosphere
        next_crossing = crossings.iloc[current_index + 1]
        if sample_end > next_crossing["start"]:
            sample_end = next_crossing["start"]

    elif row["type"] == "BS_OUT":
        sample_start = row["start"] - sample_length
        sample_end = row["start"]

        # we need to enure we don't go past into the magnetosphere
        previous_crossing = crossings.iloc[current_index - 1]
        if sample_start < previous_crossing["end"]:
            sample_start = previous_crossing["end"]

    else:
        return None


    # Load sample data:
    sample = mag.Load_Between_Dates(root_dir, sample_start, sample_end, strip=True)


    sample_position = [sample["eph_x"], sample["eph_y"], sample["eph_z"]]
    sample_average_position = np.mean([sample["eph_x"], sample["eph_y"], sample["eph_z"]], axis=1)
    # convert to radii from km
    sample_average_position /= 2439.7

    longitude = np.arctan2(sample_average_position[1], sample_average_position[0]) * 180 / np.pi

    if longitude < 0:
        longitude += 360

    local_time = ((longitude + 180) * 24 / 360) % 24

    latitude = np.arctan2(
        sample_average_position[2], np.sqrt(sample_average_position[0] ** 2 + sample_average_position[1] ** 2)
    ) * 180 / np.pi

    magnetic_latitude = np.arctan2(
        sample_average_position[2] - (479 / 2439.7), np.sqrt(sample_average_position[0] ** 2 + sample_average_position[1] ** 2)
    ) * 180 / np.pi


    return {
        # Time identifiers
        "crossing_start": row["start"],
        "crossing_end": row["end"],
        "sample_start": sample_start,
        "sample_end": sample_end,

        # Data sample itself
        "dates": sample["date"].tolist(),
        "mag_total": sample["mag_total"].tolist(),
        "mag_x": sample["mag_x"].tolist(),
        "mag_y": sample["mag_y"].tolist(),
        "mag_z": sample["mag_z"].tolist(),

        # The mean local time of the sample
        "local_time": local_time,

        # The mean latitude of the sample
        "latitude": latitude,

        # The mean magnetic latitude of the sample
        "magnetic_latitude": magnetic_latitude,
    }


# Create empty list of dictionaries
magnetosheath_samples = []

# Iterrate through the crossings
count = 0
process_items = [row for _, row in crossings.iterrows()]
with multiprocessing.Pool() as pool:
    for result in pool.imap(Get_Sample, process_items):


        if result is not None:
            # Add row dictionary to list
            magnetosheath_samples.append(result)


        count += 1
        print(f"{count} / {len(crossings)}", end="\r")


# Create dataframe from solar wind samples
magnetosheath_samples = pd.DataFrame(magnetosheath_samples)

print("")

magnetosheath_samples.to_csv(f"./total_filtered_magnetosheath_sample_{int(sample_length.total_seconds() / 60)}_mins.csv")
