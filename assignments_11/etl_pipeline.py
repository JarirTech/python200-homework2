

# ---- Project — Full ETL Pipeline

#  jarirTech

# Youtube Video: https://youtu.be/FTLCZm6HKkI

# etl_pipeline.py: a complete Prefect flow with four tasks that orchestrate the full Extract + Load + Transform + Load pipeline

# setup

import os
import json
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

import requests
from prefect import task, flow

import joblib
import pandas as pd

if load_dotenv():
    print('environment variables loaded successfuly')

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
FEATURES = metadata["feature_names"]

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)


# -----------------The Extract Task----------------------------------------
print('----The Extract task--------------------------')

# @task(retries=2, retry_delay_seconds=10)
# Calls the Open-Meteo historical archive API to fetch 2023 daily weather data for a city of your choice(Boston)
# Uses raise_for_status()
# Converts the columnar API response into a list of row dictionaries
# Prints a confirmation with the record count
# Returns the list of row dicts


import requests
from prefect import task
# Boston coordinates
LATITUDE  = 42.3601
LONGITUDE = -71.0589
# @task(retries=2, retry_delay_seconds=10)
@task(retries=2, retry_delay_seconds=10)
def extract() -> list:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   LATITUDE,
        "longitude":  LONGITUDE,
        "start_date": "2023-01-01",
        "end_date":   "2023-12-31",
        "daily": FEATURES,
        "timezone": "America/New_York",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    daily = response.json()["daily"]
    records = [
        {
            "date":               daily["time"][i],
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum":  daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
        }
        for i in range(len(daily["time"]))
    ]

    print(f"Extracted {len(records)} daily records from Open-Meteo")
    return records
#------------------------------------------------------------------------
# -------------- load_raw task-----------------------------------
print('--------------load_raw task-------------------')

# @task(retries=2, retry_delay_seconds=5)
# Upserts the raw records into weather_raw using on_conflict="date"
# Prints a confirmation with the upserted row count



@task(retries=2, retry_delay_seconds=5)
def load_raw(records: list) -> None:
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows into weather_raw")
#-----------------------------------------------------------------------------
# ---------------transform task-----------------------
print('-------------Transform task---------')
# @task
# Performs an incremental check: fetches dates already in weather_enriched and skips them
# Loads the saved sklearn Pipeline from models/weather_classifier.pkl
# Loads feature names from models/weather_classifier_metadata.json
# Runs predict and predict_proba on the unprocessed records
# Calls the OpenAI API to generate a one-sentence recommendation for each record
# Handles LLM errors gracefully with a fallback string
# Prints progress every 50 records
# Returns the complete list of enrichment records

@task
def transform(raw_records: list) -> list:
    # Performs an incremental check: fetches dates already in weather_enriched and skips them
    already_done = {
        r["date"]
        for r in supabase.table("weather_enriched").select("date").execute().data
    }
    to_process = [r for r in raw_records if r["date"] not in already_done]
    print(f"Records to transform: {len(to_process)} (skipping {len(already_done)} already enriched)")

    if not to_process:
        print("All records already enriched — nothing to do.")
        return []

    # --- ML classify ---
    clf = joblib.load("models/weather_classifier.pkl")
    df  = pd.DataFrame(to_process)
    X   = df[FEATURES]

    predictions   = clf.predict(X)
    probabilities = clf.predict_proba(X)[:, 1]

    print(f"ML classification complete. Good days: {int(predictions.sum())} / {len(predictions)}")

    enrichment_records = [
        {
            "date":             to_process[i]["date"],
            "good_for_running": bool(predictions[i]),
            "confidence":       round(float(probabilities[i]), 4),
            "llm_summary":      None,
        }
        for i in range(len(to_process))
    ]

    # --- LLM enrich ---
    for i, record in enumerate(enrichment_records):
        raw_row = to_process[i]
        prediction_text = "good for running" if record["good_for_running"] else "not ideal for running"
        user_message = (
            f"Date: {raw_row['date']}\n"
            f"High: {raw_row['temperature_2m_max']}°C, Low: {raw_row['temperature_2m_min']}°C\n"
            f"Precipitation: {raw_row['precipitation_sum']} mm\n"
            f"Max wind speed: {raw_row['wind_speed_10m_max']} km/h\n"
            f"Model prediction: {prediction_text} (confidence: {record['confidence']:.0%})"
        )
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=100,
            )
            record["llm_summary"] = response.choices[0].message.content.strip() or "Recommendation unavailable."
        except Exception as e:
            print(f"  LLM error on {record['date']}: {e}")
            record["llm_summary"] = "Recommendation unavailable."

        if (i + 1) % 50 == 0:
            print(f"  LLM enriched {i + 1} / {len(enrichment_records)} records")

    print(f"Transform complete: {len(enrichment_records)} records enriched")
    return enrichment_records



#-------------------------------------------------------------------------------------

#----------------------load_enriched task---------------------
print('-------load_enriched-------')
# @task(retries=2, retry_delay_seconds=5)
# Guards against an empty list (prints a message and returns early if nothing to load)
# Upserts enrichment records into weather_enriched using on_conflict="date"
# Prints a confirmation with the upserted row count

@task(retries=2, retry_delay_seconds=5)
def load_enriched(enrichment_records: list) -> None:
    if not enrichment_records:
        print("No new enrichment records to load.")
        return

    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows into weather_enriched")

#---------------------------------------------------------------------------------------
#-----------------Flow----------------------------
print('---------Flow------')
# @flow(log_prints=True)
# Calls all four tasks in the correct order
# Prints a final completion message

@flow(log_prints=True)
def etl_pipeline():
    raw_records        = extract()
    load_raw(raw_records)
    enrichment_records = transform(raw_records)
    load_enriched(enrichment_records)
    print("Pipeline complete.")

if __name__ == "__main__":
    etl_pipeline()


#--------------------------------------------------------------------------------------

#--------------Reflection------------------------------------

print('-------------Reflection----------')

# The pipeline ran successfully and all the Prefect tasks were completed.
# The Prefect UI showed the extract, load_raw, transform, and load_enriched tasks as completed.
# No tasks needed to retry because there were no errors during this run.
# The LLM summaries were useful because they explained if the weather was good or bad for running.
# One summary that stood out explained that rain and cold weather made the day not good for running, which matched the weather data and ML prediction.
# If I ran this pipeline every day, I would change the extract task to only get the previous day's weather instead of downloading all 365 days again.
# This would make the pipeline faster and avoid processing unnecessary data.