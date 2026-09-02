

# Jarirtech


import os
from dotenv import load_dotenv

from supabase import create_client, Client

from datetime import date



####  Part 1: Warmup  #########


# --- Supabase Connection -----------------------------------
print(" ---Supabase Connection:-----------------------------------")
# Connection Question 1----------------------------------------------------------
print(" Connection Question 1:")

#what are the two pieces of information supabase-py needs to connect to your project? 
# 
# The two pieces of information that supabase-py needs to connect to your project are the Supabase URL and the Supabase Key.
# 
# 
# Where do you find them in the Supabase dashboard?
# 
# On supabase dashboard on the left under settings, click on API and you will find the URL and the Key.
# 
# 
# 
# Why should they never be hardcoded in a Python script?
#
# These credentials should never be hardcoded in a Python script because they are sensitive information that can be used by automated systems
# in minutes if committed to a public repository.

#-----Connection Question 2------------------------------------------------------
print(" ---Connection Question 2--------------------------------")

load_dotenv()  # reads .env and sets environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
def get_client():
    """
    Returns a supabase client using the URL and KEY from environment variables.
    """
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Environment variables 'SUPABASE_URL' and 'SUPABASE_KEY' must be set.")
    return supabase


#----Connection Question 3------------------------------------------------------
print(" ---Connection Question 3--------------------------------")

### what is Row Level Security (RLS)?
#
# RLS lets you define fine-grained access policies — for example, "a user can only read their own rows."
# It is an important production feature, but it adds complexity during development
#
### why did you disable it on your tables for this course?
#
# We disabled RLS on our tables for this course to simplify the development process and avoid potential access issues during testing and learning.


### ---supabase-py CRUD--------------------------------------------------

print(" ---supabase-py CRUD-----------------------------------")
### ---CRUD Question 1------------------------------------------------------
print(" ---CRUD Question 1-----------------------------------")

# Write a function insert_test_record(supabase) that inserts a single row into weather_raw with today's date 
# and plausible values for all four weather columns.

def insert_test_record(supabase: Client):
    """
    Inserts a record into the 'weather_raw' table.
    """

    record = {
    "date":    date.today().isoformat(),
    "temperature_2m_max": 22.3,
    "temperature_2m_min": 14.1,
    "precipitation_sum":  0.0,
    "wind_speed_10m_max": 18.5,
    }
    response = supabase.table("weather_raw").insert(record).execute()
    print('inserted one record into weather_raw')
    return response.data

result = insert_test_record(supabase=get_client())

print(result)

### what would happen if you ran the function twice? 
# 
# If I run the function more than one time I will get an error message because the date column is unique and it's a primary key. 
# The second time I run the function it will try to insert a record with the same date and it will violate the unique constraint of the primary key.
# 
# How would you change the call to make it safe to run multiple times?
#
# I will use upsert instead of insert. Upsert will insert a new record if it doesn't exist, or update the existing record if it does exist.

###----CRUD Question 2------------------------------------------------------
print(" ---CRUD Question 2-----------------------------------")

def get_records_by_date_range(supabase, start, end):
    """
    Returns all records from the 'weather_raw' table between the given start and end dates.
    """
    response = supabase.table("weather_raw").select("*").gte("date", start).lte("date", end).execute()
    print(f"the result: {response.data}")
    return response.data

get_records_by_date_range(supabase=get_client(), start="2026-01-01", end="2026-12-31")

##-----CRUD Question 3------------------------------------------------------
print(" ---CRUD Question 3-----------------------------------") 

# insert will try to insert a new record and it will fail if the record already exists.
# upsert let you insert a record safely even the record already exists. It will update the existing record
# if it does exist, or insert a new record if it doesn't exist.

## example of insert: 
# I would use insert when adding a new student to a students table.
# Each student is a new record, so I do not want to replace an
# existing student.

## example of upsert:
# I would use upsert when updating a student's information or saving a student's grade in a students table.

def safe_upsert(supabase, records):
    """
    Upserts a list of records into the 'weather_raw' table.
    """
    response = supabase.table("weather_raw").upsert(records, onConflict="date").execute()
    print(f"the number of rows affected: {len(response.data)}")
    return response.data

##----Idempotency------------------------------------------
#-----------Idempotency Question 1--------------------------------------
print(" ---Idempotency----------------------------------")
print(" ---Idempotency Question 1-----------------------------------")

## "Idempotency"  
# Explain why idempotency matters for a data pipeline.
# 
# Idempotency is important in a data pipeline because it ensures that running the same operation multiple times has the same effect as running it once.
#  This is ensure the consistency of the data and its reliability. 
# 
#  Give one concrete example of what goes wrong in a non-idempotent pipeline when the script crashes halfway through and is restarted.
#
# An example of what goes wrong in a non-idempotent pipeline is if a script crashes halfway through and is restarted, 
# it may insert duplicate records into the database, leading to data inconsistencies and errors in downstream processes that rely on that data.
# 
# For example, a pipeline adds student records to a students table.
# It adds 30 students and then crashes before finishing.

# If we restart the pipeline using insert, the first 30 students
# could be added again and create duplicates.

# If we use upsert with student_id as a unique key, the existing
# students are updated instead of duplicated. This makes the
# pipeline safe to run multiple times.