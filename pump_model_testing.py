"""
Example script for working with the pump model. Here we retrieve MESSENGER's trajectory for a given time and pull the expected field data from the model. We then compare the model to the measured field.
"""

import datetime as dt
import sys
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import spiceypy as spice
import hermpy.mag as mag
import hermpy.trajectory as trajectory

pump_directory = "/home/daraghhollman/Main/mercury/KTH22-model/"
pump_control_params = pump_directory + "control_params_v8b.json"
pump_fit_params = pump_directory + "kth_own_cf_fit_parameters_opt_total_March23.dat"
sys.path.append(pump_directory)

from kth22_model_for_mercury_v8 import kth22_model_for_mercury_v8 as Pump

mpl.rcParams["font.size"] = 14
root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"
metakernel = "/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt"
spice.furnsh(metakernel)

# Select disturbance index. Here we assume the mean value of 50
disturbance_index = 50

# Define a region of time to query
start_time = dt.datetime(year=2011, month=7, day=27, hour=20, minute=0)
end_time = dt.datetime(year=2011, month=7, day=27, hour=22, minute=0)


# Load MAG data
if (end_time - start_time).days == 0:
    dates_to_load = [start_time]

else:
    dates_to_load: list[dt.datetime] = [
        start_time + dt.timedelta(days=i) for i in range((end_time - start_time).days)
    ]

files_to_load: list[str] = []
for date in dates_to_load:
    file: list[str] = glob(
        root_dir
        + f"{date.strftime('%Y')}/*/MAGMSOSCIAVG{date.strftime('%y%j')}_01_V08.TAB"
    )

    if len(file) > 1:
        raise ValueError("ERROR: There are duplicate data files being loaded.")
    elif len(file) == 0:
        raise ValueError("ERROR: The data trying to be loaded doesn't exist!")

    files_to_load.append(file[0])

data = mag.Load_Messenger(files_to_load)
data = mag.Strip_Data(data, start_time, end_time)

# Load ephemeris data for that time
positions = trajectory.Get_Trajectory("MESSENGER", [start_time, end_time], steps=int((end_time - start_time).total_seconds()) + 1)

# Here our example is around only an hour long, we can assume a constant heliocentric distance
midpoint = start_time + (end_time - start_time) / 2
heliocentric_distance = trajectory.Get_Heliocentric_Distance(midpoint)

# Convert to AU
heliocentric_distance /= 1.496e+8

# Determine the field for the trajectory
pump_field = Pump(positions[:,0], positions[:,1], positions[:,2], heliocentric_distance, disturbance_index, pump_control_params, pump_fit_params)

print(pump_field)


plt.plot(data["date"], data["mag_x"])
plt.plot(data["date"], pump_field[0])
plt.show()
