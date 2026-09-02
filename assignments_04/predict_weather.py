# predict_weather.py
# By JarirTech

import json
import joblib
import pandas as pd



#Task 1: Load and Verify
print("*************** Task 1: Load and Verify ***************")

model_path = "models/weather_classifier.pkl"
model = joblib.load(model_path)

#Load the metadata
with open("models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)


print("Model & metadata loaded successfully")
print("City:", metadata["city"]["name"])
print("Latitude:", metadata["city"]["latitude"])
print("Longitude:", metadata["city"]["longitude"])
print()

print("Features:")
for feature in metadata["feature_names"]:
    print("-", feature)
print()
print("Test AUC:", metadata["test_auc"])

#************************************************************

#** Task 2: Predict on New Data***************
print("*************** Task 2: Predict on New Data ***************")

new_days = pd.DataFrame(
    [
        # Nice spring day
        [20, 10, 0.0, 15],

        # Cold and snowy day
        [-8, -12, 2.0, 20],

        # Hot summer day
        [32, 22, 0.0, 18],

        # Rainy and windy day
        [18, 12, 8.0, 40],

        
        # Borderline day
        [7, 0, 2.9, 29],
    ],
    columns=metadata["feature_names"],
)

predictions = model.predict(new_days)
probabilities = model.predict_proba(new_days)
for i in range(len(new_days)):
    print(f"\nDay {i+1}")

    print("Temperature Max:", new_days.iloc[i]["temperature_2m_max"])
    print("Temperature Min:", new_days.iloc[i]["temperature_2m_min"])
    print("Precipitation:", new_days.iloc[i]["precipitation_sum"])
    print("Wind Speed:", new_days.iloc[i]["wind_speed_10m_max"])

    if predictions[i] == 1:
        label = "Good"
    else:
        label = "Skip"

    print("Prediction:", label)
    print("Probability of good running day:",
          round(probabilities[i][1], 2))

#************************************************************
#*******Task 3: Reflect*********
print("\n*************** Task 3: Reflect ***************")

#1. The borderline case is Day 5 because its weather values are close to the
# limits I used to define a good running day. The model predicted "Skip"
# with a probability of 0.13 for a good running day, so it seems confident
# that this is not a good day for running. If the model predicted a probability
# of 0.52, I would consider that an uncertain prediction because it is very
# close to 0.5. In that case, I might show a message to the user saying that
# the weather is borderline and let them decide. 
# 2. If someone runs predict_weather.py before train_weather_classifier.py,
#  the model and metadata
# files will not exist, so the program will give a FileNotFoundError. A more
# helpful error message would tell the user to run train_weather_classifier.py first
# 3. In a real application, I would replace the manually created weather
# data with tomorrow's weather forecast from the Open-Meteo API so the model
# could automatically predict whether the next day is good for running or not.