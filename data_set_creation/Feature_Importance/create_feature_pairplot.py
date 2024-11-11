import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

combined_features = pd.read_csv("/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv") 
# print(combined_features.columns)

combined_features["RH"] = combined_features["RH"] / 1.496e+8

# Filter by LT
# combined_features = combined_features.loc[ combined_features["LT"].between(11, 13) ]

# Filter by Heliocentric Distance
filtered_combined_features = combined_features.loc[ combined_features["RH"].between(0.42, 0.47) ]

# Filter outliters
filtered_combined_features = filtered_combined_features[(scipy.stats.zscore(combined_features.select_dtypes(include='float64')) < 3).all(axis=1)]

print(f"N={len(filtered_combined_features)}")

# Create pairplots
sns.pairplot(filtered_combined_features, vars=["median |B|", "std |B|", "std Bx", "mean By", "std Bz"], hue="label", corner=True, kind="kde", plot_kws=dict(alpha=0.5))
plt.show()
