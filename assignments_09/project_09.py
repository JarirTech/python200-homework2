# Project — Extract + Load Pipeline
# JarirTech

# Video: https://youtu.be/krRhDVg9K0I

import os
from datetime import date

import requests
from dotenv import load_dotenv
from supabase import create_client


# --- Step 1: Extract -----------------------------------
print(" ---Step 1: Extract-----------------------------------")

# Boston weather data for the full year 2023.
url = "https://archive-api.open-meteo.com/v1/archive"

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
    "temperature_unit": "celsius",
    "precipitation_unit": "mm",
    "wind_speed_unit": "kmh",
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

data = response.json()

print("Data extracted successfully from the API.")
print("City: Boston")
print("Days received:", len(data["daily"]["time"]))
print("Units:", data["daily_units"])

# Check that these units match the Week 4 training data.


# --- Step 2: Transform -----------------------------------
print(" ---Step 2: Transform-----------------------------------")

# Convert the separate lists into one dictionary for each day.
records = []
daily = data["daily"]

for i in range(len(daily["time"])):
    row = {
        "date": daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum": daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
    }

    records.append(row)

print("The number of records is:", len(records))

if not records:
    raise ValueError("The API returned no daily weather records.")

print("First record:", records[0])
print("Last record:", records[-1])

# I expected 365 records because 2023 was not a leap year.
# I received 365 records, so the numbers match.
# If the numbers differed, I would check the requested dates
# and whether the API response was incomplete.


# --- Step 3: Load -----------------------------------
print(" ---Step 3: Load-----------------------------------")

# Call load_dotenv() only once.
if not load_dotenv():
    print("No environment variables were loaded from the .env file.")
else:
    print("Environment variables loaded successfully.")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Credentials may also be set outside the .env file.
# Check them before creating the client.
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "Environment variables 'SUPABASE_URL' and 'SUPABASE_KEY' must be set."
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# The date column must be unique or the primary key.
response = (
    supabase.table("weather_raw")
    .upsert(records, on_conflict="date")
    .execute()
)

print(f"Successfully upserted {len(records)} weather records.")

# Upsert adds a row if its date does not exist.
# If the date already exists, it updates that row.
# Loading the same records again does not create duplicate rows.
# This makes the load idempotent.

# Run this updated script twice and fill in your results:
# Total rows after the first run: ____
# Total rows after the second run: ____
# Did the total stay the same? ____


# --- Step 4: Verify -----------------------------------
print(" ---Step 4: Verify-----------------------------------")

# Print the total number of rows in the whole table.
count_response = (
    supabase.table("weather_raw")
    .select("date", count="exact", head=True)
    .execute()
)

print(f"Total records in weather_raw: {count_response.count}")


# Find the earliest date: sort from oldest to newest.
first = (
    supabase.table("weather_raw")
    .select("date")
    .order("date")
    .limit(1)
    .execute()
)

# Find the latest date: sort from newest to oldest.
last = (
    supabase.table("weather_raw")
    .select("date")
    .order("date", desc=True)
    .limit(1)
    .execute()
)

if first.data and last.data:
    print(f"Earliest date in weather_raw: {first.data[0]['date']}")
    print(f"Latest date in weather_raw: {last.data[0]['date']}")
else:
    print("The table is empty.")


# Find the row for July 4.
target_date = "2023-07-04"

july_4 = (
    supabase.table("weather_raw")
    .select("*")
    .eq("date", target_date)
    .execute()
)

if july_4.data:
    print(f"Row for 2023-07-04: {july_4.data[0]}")

else:
    # Find the closest available date before July 4.
    before = (
        supabase.table("weather_raw")
        .select("*")
        .lt("date", target_date)
        .order("date", desc=True)
        .limit(1)
        .execute()
    )

    # Find the closest available date after July 4.
    after = (
        supabase.table("weather_raw")
        .select("*")
        .gt("date", target_date)
        .order("date")
        .limit(1)
        .execute()
    )

    # Compare the two possible records.
    candidates = before.data + after.data

    nearest_row = None
    smallest_difference = None
    target = date.fromisoformat(target_date)

    for row in candidates:
        row_date = date.fromisoformat(row["date"])

        # Find the number of days between the two dates.
        difference = abs((row_date - target).days)

        if smallest_difference is None or difference < smallest_difference:
            smallest_difference = difference
            nearest_row = row

    if nearest_row is not None:
        print("July 4 is missing. Nearest record:", nearest_row)
    else:
        print("No weather records were found.")