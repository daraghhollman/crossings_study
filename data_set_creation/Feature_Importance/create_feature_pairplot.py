import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import scipy.stats

combined_features = pd.read_csv("./combined_features.csv") 

# Filter outliters
filtered_combined_features = combined_features[(scipy.stats.zscore(combined_features.select_dtypes(include='float64')) < 3).all(axis=1)]

# Create pairplots
seaborn.pairplot(filtered_combined_features, vars=["median |B|", "std |B|", "std Bx", "std By", "std Bz", "x_msm"], hue="label", corner=True, kind="kde", plot_kws=dict(alpha=0.5))
plt.show()
