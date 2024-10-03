"""
Observe and compare distribution of data for categorical regression
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

messenger = pd.read_csv("./region_data_5_mins.csv")[["region", "mag_total", "mag_x", "mag_y", "mag_z"]]

print(f"Looking at {len(messenger)} rows")

sns.pairplot(messenger, hue="region");
plt.show()
