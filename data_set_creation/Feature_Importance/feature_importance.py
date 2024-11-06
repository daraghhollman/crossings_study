import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

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

cm_display = ConfusionMatrixDisplay(cm, display_labels=["Magnetosheath", "Solar Wind"])
cm_display.plot()
plt.show()
