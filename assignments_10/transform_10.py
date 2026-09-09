
###---------------------------Part 2: Project — The Double-Transform Pipeline------------------------------
# jarirTech

### video: https://youtu.be/yVKTXIihw0Y 

import os
from dotenv import load_dotenv
from openai import OpenAI

import supabase
from supabase import create_client

import json
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta

from pathlib import Path

if load_dotenv():
    print("Environment variables loaded from .env file.")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
#---------------------------------------------------------------------------------
###  ---Step 1: Incremental Read----
print(" ---Step 1: Incremental Read-----------------------------------")
# Loading metadata model
# 
# path for the ML model and metadata 

# Metadata path
metadata_path = Path("models/weather_classifier_metadata.json")

# Model path
clf_path = Path("models/weather_classifier.pkl")

with open(metadata_path, "r") as f:
    metadata = json.load(f)
print("Metadata loaded successfully.")
#print(metadata)

# Fetch all rows from weather_raw. Fetch all dates already present in weather_enriched. Determine which records still need processing.

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
weather_raw = supabase.table("weather_raw").select("*").execute().data
weather_enriched = supabase.table("weather_enriched").select("*").execute().data
print(f"Fetched weather_raw records: {len(weather_raw)}")
print(f"Fetched weather_enriched records: {len(weather_enriched)}")

# Find dates already processed
enriched_dates = {row["date"] for row in weather_enriched}


# Keep only records that still need processing
records_to_process = [
    row for row in weather_raw
    if row["date"] not in enriched_dates
]

print("Records needing processing:", len(records_to_process))
# Stop if all records are already enriched
if len(records_to_process) == 0:
    print("No new records to process. All weather records are already enriched.")
else:
    # All records in weather_enriched needs to be processed

    #------------------------------------------------------------------------------------------------------------
    #------------------Step 2: ML Transform-----------------
    print(" ---Step 2: ML Transform-----------------------------------")

    # Loading the ML model

    

    clf = joblib.load(clf_path)
    print("ML model loaded successfully.")

    #  Building a DataFrame

    df = pd.DataFrame(records_to_process)
    print("DataFrame created successfully.")
    #print(df.head())
    # df columns :
    #  date  temperature_2m_max  temperature_2m_min  precipitation_sum  wind_speed_10m_max                  loaded_at
    # metadata columns

    # 'feature_names': ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max'],


    features = metadata["feature_names"]
    print("Features:", features)

    ### predict and predict_proba

    X = df[features]
    predict = clf.predict(X)
    predict_proba = clf.predict_proba(X)

    # print('Predictions:', predict)
    # print('Prediction Probabilities:', predict_proba)
    ## Build a list of enrichment records with date, good_for_running, and confidence
    # Build a list of enrichment records
    enrichment_records = []

    for i, row in df.iterrows():
        enrichment_record = {
        "date": row["date"],
        "good_for_running": bool(predict[i]),
        "confidence": float(predict_proba[i][predict[i]])
    }
        

        enrichment_records.append(enrichment_record)

    # summary of the predictions: 
    predicted_good = sum(predict)
    print(f"Predicted good days: {predicted_good}")
    # confidence range
    confidences = [record["confidence"] for record in enrichment_records]

    print("Confidence range:", f"{min(confidences):.4f}", "to", f"{max(confidences):.4f}")
    #print(enrichment_records[:5])  # Print the first 5 enrichment records for verification

    # [{'date': '2023-01-01', 'good_for_running': 0, 'confidence': 0.9644008822224853}, 
    # {'date': '2023-01-02', 'good_for_running': 0, 'confidence': 0.5155959853523282}, 
    # {'date': '2023-01-03', 'good_for_running': 0, 'confidence': 0.9954837286651362},
    #  {'date': '2023-01-25', 'good_for_running': 0, 'confidence': 0.917965237232752}, 
    # {'date': '2023-02-05', 'good_for_running': 0, 'confidence':0.8820608403247574}]

    # print('length of enrichment records:', len(enrichment_records))
    #length of enrichment records: 365

    #--------------------------------Step 3: LLM Transform-----------------
    print(" ---Step 3: LLM Transform-----------------------------------")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


    # System prompt for the LLM
    SYSTEM_PROMPT = (
        "You are writing a one-sentence running recommendation for a daily weather summary app. "
        "You will receive weather conditions for a single day and a machine learning prediction "
        "about whether the day is good for running. "
        "Write exactly one sentence that is direct, practical, and specific to the conditions. "
        "Do not use bullet points, headers, or phrases like 'Based on the data'."
    )


    # User message function
    def make_user_message(row, good_for_running, confidence):

        prediction_text = (
            "good for running"
            if good_for_running
            else "not ideal for running"
        )

        return (
            f"Date: {row['date']}\n"
            f"High: {row['temperature_2m_max']}°C, "
            f"Low: {row['temperature_2m_min']}°C\n"
            f"Precipitation: {row['precipitation_sum']} mm\n"
            f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
            f"Model prediction: {prediction_text} "
            f"(confidence: {confidence:.0%})"
        )


    # Check that the LLM returned one sentence
    # def validate_summary(text):

    #     text = text.strip()

    #     if not text:
    #         return None

    #     sentences = [s for s in text.split(".") if s.strip()]

    #     if len(sentences) != 1:
    #         return None

    #     return text


    # Process every enrichment record
    for i, record in enumerate(enrichment_records):  

        raw_row = next(
            r for r in records_to_process
            if r["date"] == record["date"]
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": make_user_message(
                            raw_row,
                            record["good_for_running"],
                            record["confidence"]
                        )
                    }
                ],
                max_tokens=100
            )

            summary = response.choices[0].message.content.strip()

        except Exception as e:
            print(f"API error on {record['date']}: {e}")
            summary = "Recommendation unavailable."

        # Add the LLM result to the existing dictionary
        record["llm_summary"] = summary

        # Print progress every 50 records
        if (i + 1) % 50 == 0:
            print(
                f"Enriched {i + 1} / "
                f"{len(enrichment_records)} records..."
            )

    #--------------------------------Step 4: Load to Supabase-----------------
    print(" ---Step 4: Load to Supabase-----------------------------------")    


    # Upserting all enrichment records into weather_enriched

    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows into weather_enriched")


# ------Step 5: Verify------------------------------
print(" ---Step 5: Verify-----------------------------------")

# Query weather_enriched and print:
# The total number of rows
# Five sample rows showing date, good_for_running, confidence, and llm_summary
# The number of days classified as good for running

# Count total rows
total_rows = (supabase.table("weather_enriched").select("date", count="exact").execute())

print(f"Total rows in weather_enriched: {total_rows.count}")

# Get five sample rows
check = (supabase.table("weather_enriched").select("*").limit(5).execute())

print("\nFive sample rows:")

for row in check.data:
    print(
        f"{row['date']} | "
        f"good={row['good_for_running']} | "
        f"conf={row['confidence']:.2f}"
    )
    print(f"  {row['llm_summary']}")
    print()

# Count how many were classified as good
good_count = (supabase.table("weather_enriched").select("date", count="exact").eq("good_for_running", True).execute())

print(f"Good-for-running days in weather_enriched: {good_count.count}")


# Reflection:
# Most of the LLM summaries match the weather data and the model prediction.
# January 3 is a good example because the summary mentions the cold temperature
# and rain and clearly recommends skipping the run.
# January 2 seems a little off because the weather was mild with no precipitation,
# but the model still predicted that it was not ideal for running.
# The model confidence was only about 52%, so the model was not very sure.
# Since the LLM uses the ML prediction, an uncertain prediction can lead to
# a weaker recommendation.

# ------Step 6: Reflect------------------------------

# The assignment says the ML classifier was trained on Charlotte, NC weather data,
# But my ML model was trained on Boston weather data, and my Week 9 data is also from Boston.
# Because the data is from the same city, I expect the predictions to be more accurate.
# The LLM does not change the ML prediction; it only adds a simple recommendation.
# This means if the ML prediction is wrong, the LLM may also give a bad recommendation.
# With 50,000 records, my main concerns would be the API cost and the time it takes.
# I could process the records in smaller batches and only process new records to save time and money.