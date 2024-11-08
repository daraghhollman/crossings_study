import pandas as pd

# Load the feature datasets
sw_features = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/crossings_study/data_set_creation/solar_wind_features.csv"
)
ms_features = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/crossings_study/data_set_creation/magnetosheath_features.csv"
)

features = ["mean", "median", "std", "skew", "kurtosis", "dip_stat", "dip_p_value", "grazing_angle", "RH", "LT", "Lat", "MLat", "x_msm", "y_msm", "z_msm", "is_inbound"]
expanded_feature_labels = ["|B|", "Bx", "By", "Bz"]

# Select only the columns we want to keep
sw_features = sw_features[features].copy()
ms_features = ms_features[features].copy()

# Process each dataset
for dataset in [sw_features, ms_features]:
    for feature in features[0:7]:

        # Convert elements from list-like strings to lists of floats
        dataset[feature] = dataset[feature].apply(
            lambda s: list(map(float, s.strip("[]").split()))
        )

        # Expand feature lists into new columns
        expanded_columns = dataset[feature].apply(pd.Series).rename(
            lambda x: f"{feature} {expanded_feature_labels[x]}", axis=1
        )
        
        # Assign new columns back to the original dataset
        dataset[expanded_columns.columns] = expanded_columns

    # Drop original feature columns
    dataset.drop(columns=features[0:7], inplace=True)

sw_features["label"] = "Solar Wind"
ms_features["label"] = "Magnetosheath"

combined_features = pd.concat([sw_features, ms_features], ignore_index=True)

combined_features.to_csv("./combined_features.csv")
