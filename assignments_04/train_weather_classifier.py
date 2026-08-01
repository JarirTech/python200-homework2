import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, RocCurveDisplay

import joblib
import json
import sys
import sklearn


import os
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# Step 1: Fetch the Data
print('***************Step 1: Fetch the Data***************')
url = "https://archive-api.open-meteo.com/v1/archive"
# boston weather data for 2023
params = {
    "latitude": 42.3601,
    "longitude": -71.0589,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()  # Raise an error for bad responses
data = response.json()  
df = pd.DataFrame(data["daily"])

# explore the data
df.shape  # checking the shape of the DataFrame
df.info()  # checking the info of the DataFrame
df.describe()  # checking the summary statistics of the DataFrame
df.isnull().sum()  # checking for missing values in the DataFrame
df.duplicated().sum()  # checking for duplicate rows in the DataFrame
df.columns  # checking the column names of the DataFrame
df.dtypes  # checking the data types of the columns in the DataFrame

print(df.head())  # checking the first few rows of the DataFrame

# Preprocess the Data

# Convert the 'time' column to datetime format as was str
df["date"] = pd.to_datetime(df["time"])

# drop the original 'time' column
df = df.drop("time", axis=1)

#**********************************************************************

## *******Step 2: Engineer Labels****

print('***************Step 2: Engineer Labels***************')

# I used same thresholds also for Boston Ma

df["good_running"] = (
    (df["temperature_2m_max"] >= 7)
    & (df["temperature_2m_max"] <= 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int) 

print(df["good_running"].value_counts())  # checking the distribution of the target variable
good_fraction = df["good_running"].mean()

print()
print("Fraction of good running days:", round(good_fraction, 2))

# About 38% of the days were labeled as good for running.
# I think this is reasonable because here in Boston we have a cold winters,
# warm summers, and some rainy or windy days. These weather
# conditions mean that many days are not ideal for running.

#**********************************************************************
# ********Step 3: Train and Tune******
print('***************Step 3: Train and Tune***************')

features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[features]
y = df["good_running"]

# splitting the data into training and test sets, stratifying by the target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# pipeline 
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    "clf__C": [0.1, 1, 10, 100, 1000]
}


# Using GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc")
grid_search.fit(X_train, y_train)

# Print the best C value and best CV AUC
print("Best C value:", grid_search.best_params_["clf__C"])
print("Best CV AUC:", round(grid_search.best_score_, 2))

# best model
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Print a full classification report on the test set
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print the test AUC
test_auc = roc_auc_score(y_test, y_pred_proba)
print("Test AUC:", round(test_auc, 2))

#plot 
RocCurveDisplay.from_predictions(y_test, y_pred_proba, name=f"Logistic Regression (AUC={test_auc:.3f})")
plt.title("Weather Classifier ROC Curve")
plt.savefig("outputs/weather_roc.png")
plt.close()

print("ROC curve saved to outputs/weather_roc.png")

#**********************************************************************
#***Step 4: Reflect on Evaluation*******
print('***************Step 4: Reflect on Evaluation***************')

# The AUC score of 0.75 means the model does a fairly good job of
# separating good running days from bad running days. It is not
# perfect, but it performs better than random guessing. Looking at
# the classification report, the recall for the "good running" class
# is lower than the precision, which means the model misses some days
# that are actually good for running (false negatives). I would rather
# miss a few good running days than recommend running on a day with
# bad weather. If I were building a real app, I might keep the threshold
# close to 0.5 or increase it a little to make the recommendations more
# careful.

#**********************************************************************
#****Step 5: Save the Model*******  
print('***************Step 5: Save the Model***************')


# Save the trained pipeline
model_path = "models/weather_classifier.pkl"

joblib.dump(
    best_model,
    model_path
)


metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "feature_names": features,
    "best_hyperparameters": grid_search.best_params_,
    "test_auc": test_auc,
    "city": {
        "name": "Boston, MA",
        "latitude": 42.3601,
        "longitude": -71.0589
    },
    "label_thresholds": {
    "temperature_2m_max": "7 to 26 °C",
    "temperature_2m_min": ">=  0 °C",
    "precipitation_sum": "< 3 mm",
    "wind_speed_10m_max": "< 30 km/h"
    }

    }

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f,  indent=4)

print("Model and metadata saved to models/ directory.")
