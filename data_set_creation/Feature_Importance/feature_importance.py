import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


show_plots = False

# Load data
combined_features = pd.read_csv("./combined_features.csv")

X = combined_features.drop(columns=["label"])  # Features

# Isolate important features
X = X.drop(
    columns=[
        "skew Bx",
        "skew By",
        "skew Bz",
        "kurtosis By",
        "kurtosis Bz",
    ]
)
X = X.iloc[:, 1:]  # Remove the index column
y = combined_features["label"]  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train, y_train)

if show_plots:

    y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    importances = random_forest.feature_importances_
    feature_names = X.columns

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title("Feature Importance")
    plt.show()

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix\n")
    print(cm)

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    cm_display = ConfusionMatrixDisplay(
        cm, display_labels=["Magnetosheath", "Solar Wind"]
    )
    cm_display.plot()
    plt.show()


if input("Save predictions to csv? [Y/n]\n > ") != "n":
    truths = combined_features["label"].tolist()  # What the correct label is
    predictions = []  # What the random forest predicted
    sheath_probability = []  # probability it is sheath
    solar_wind_probability = []  # probability it is solar wind

    # Create dataframe of how well the random forest performed
    for i in tqdm(range(len(X))):
        sample = X.iloc[i].to_frame().T

        prediction = random_forest.predict(sample)[0]
        probabilities = random_forest.predict_proba(sample)

        predictions.append(prediction)
        sheath_probability.append(probabilities[0][0])
        solar_wind_probability.append(probabilities[0][1])

    prediction_data = pd.DataFrame(
        {
            "Truth": truths,
            "Prediction": predictions,
            "P(Magnetosheath)": sheath_probability,
            "P(Solar Wind)": solar_wind_probability,
        }
    )
    prediction_data.to_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/random_forest_predictions.csv"
    )
