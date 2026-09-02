
# Project — Extract + Load Pipeline
# JarirTech

##Video: https://youtu.be/krRhDVg9K0I



import os
from dotenv import load_dotenv
import pandas as pd

from supabase import create_client, Client

from datetime import date

import requests
import json

### ------Step 1: Extract------------------------------

url = "https://archive-api.open-meteo.com/v1/archive"
# Boston weather data for 2023

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
#df= pd.DataFrame(data["daily"]) 
print("Data extracted successfully from the API.")
#print(df.head())  # Print the first few records to verify

###----Step 2: Transform-----------------------------------
print(" ---Step 2: Transform-----------------------------------")


# weather_raw_columns: date, temperature_2m_max, temperature_2m_min, precipitation_sum, wind_speed_10m_max, loaded_at
# data_columns:     time  temperature_2m_max  temperature_2m_min  precipitation_sum  wind_speed_10m_max

# Convert the API response from columnar format into a list of row dictionaries. Each dictionary should have keys that 
# exactly match the column names in weather_raw.
records = []
daily = data["daily"]
for i in range(len(daily["time"])):
    row = {
        "date": daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum": daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
        "loaded_at": date.today().isoformat()
    }
    records.append(row)

print(f"The number of records is:  {len(records)}")
#Print the first and last record to confirm the transformation looks correct. 


# First record:
print("First record:", records[0])
# Last record:
print("Last record:", records[-1])


# Add a comment: how many records do you expect for a full year, and how many did you get? If the numbers differ,
#  what might explain the discrepancy?

## I am expecting 365 records for a full year, and I got 365 records. The numbers match the expectation. 

##-----Step 3: Load-----------------------------------
print(" ---Step 3: Load-----------------------------------")

load_dotenv()  # reads .env and sets environment variables
if load_dotenv():
    print("Environment variables loaded successfully.")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Environment variables 'SUPABASE_URL' and 'SUPABASE_KEY' must be set.")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

response = (
    supabase.table("weather_raw")
    .upsert(records, on_conflict="date")
    .execute()
)

print(f"Upserted {len(response.data)} rows into weather_raw")

#Run the script a second time and confirm the row count in weather_raw does not change. 
# Add a comment: what does this tell you about idempotency?

## I got the same number of records after running the script a second time, which confirms that the upsert operation is idempotent.
#  This means that running the same operation multiple times will not change the result beyond the initial application, 
# ensuring data consistency and preventing duplicate entries.

###-----Step 4: Verify-----------------------------------------------

print(" ---Step 4: Verify-----------------------------------")

## Prints the total number of rows in weather_raw
count_response = supabase.table("weather_raw").select("date", count = 'exact').execute()
print(f"Total records in weather_raw: {count_response.count}")


## Prints the earliest and latest dates in the table
# First date
first = supabase.table("weather_raw").select("*").eq("date", "2023-01-01").execute()

# Last date
last = supabase.table("weather_raw").select("*").eq("date", "2023-12-31").execute()


print(f"Earliest date in weather_raw: {first.data[0]['date']}")
print(f"Latest date in weather_raw: {last.data[0]['date']}")


## Prints the row for 2023-07-04
july_4 = supabase.table("weather_raw").select("*").eq("date", "2023-07-04").execute()
print(f"Row for 2023-07-04: {july_4.data[0]}")
