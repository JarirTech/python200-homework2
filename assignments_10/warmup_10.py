#jarirTech


from supabase import create_client, Client
from openai import OpenAI

from time import sleep


##---Part 1: Warmup-----

###---- ML vs. LLM in Pipelines------

## In a comment block, explain the difference between what the ML classifier produces 
# and what the LLM produces in this week's pipeline

# In this week pipeline the ML and LLM both are part of the Transform step.

#  the ML classifier produces a binary label and a confidence score for each day. It uses the weather numeric data to make a binary prediction,

# such as if the weather is good for running or not. It is fast, cheap, and gives reproducible results.


# The ML model is better for the binary prediction because it is fast and consistent.

# The LLM is better for writing a recommendation because it understands and generates language.

# If we swap them, the LLM may give different predictions, and the ML model cannot write

# a natural-language recommendation.

###----ML/LLM Question 2----------

##Converting a date string like "2023-07-04" to day-of-week
# I will use determinstic code to convert a date string to a day-of-week. because it will always give the same result for the same input.

##Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text
# I will use an LLM to classify a job posting because it can understand and generate language, and it can handle the variability in freeform text.

## Predicting customer churn given 15 numeric features and a labeled training dataset
# I will use a ML model to predict customer churn because I can train the model on the labeled training dataset and it can learn patterns in the 
# numeric features to make predictions.

## Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form
# I will use an LLM to normalize inconsistent city names because it can understand and generate language, and it can handle the variability in freeform text.


## Summing a column of revenue figures
# I will use a deterministic function to sum a column of revenue figures because it will always give the same result for the same input,
# and it can be implemented with a simple aggregation operation.

## ------ML/LLM Question 3----------

# what is incremental processing and why is it important for this pipeline?

# Incremental processing means only process the data that has changed since the last time the pipeline ran. It is more efficient than reprocessing 
# all the data every time, especially when the dataset is large and only a small portion of it changes between runs like in this pipeline.

# What would happen — in terms of cost and data correctness — if the transform script re-processed all 365 records every time it ran?
# There will be more cost not necessary because the ML model and LLM will be called for all 365 records, even if only a few have changed. Also it could lead
#  to data correctness issues if the ML model or LLM produces different results for the same input on different runs, which could happen if they are not deterministic.


###---Prompt Design------------------------------------------------------------------------------------------------------------

# ----------Prompt Question 1------------------------------

#SYSTEM_PROMPT = (
#     "You are writing two sentences:"
#     "First sentence states the prediction: good for running or skip."
#     " Second sentence explains the reasoning for a daily weather summary app. "
#     "You will receive weather conditions for a single day and a machine learning prediction "
#     "about whether the day is good for running. "
#     "Write exactly two sentences — direct, practical, and specific to the conditions. "
#     "Do not use bullet points, headers, or phrases like 'Based on the data'."
# )
##  what would you need to change in the validation logic to accommodate two sentences instead of one?

# I will check that the output contains exactly two sentences, and that each sentence is complete and makes sense in the context 
# of the weather conditions and the ML prediction. I will also check that the first sentence states the prediction and the second 
# sentence explains the reasoning, as specified in the prompt.


###----------Prompt Question 2------------------------------

def call_with_retry(client, messages, max_retries=3):
    """
     calls client.chat.completions.create() and retries up to max_retries times on any exception, with a 2-second wait between attempts.
     On final failure, return None
    """
    max_attempts = max_retries + 1
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                
            )
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                print("Retrying in 2 seconds...")
                sleep(2)
            else:
                print("Max retries reached. Returning None.")
                return None

# I will use this function in a production pipeline to handle transient errors when calling the OpenAI API. 
# It will help ensure that temporary issues do not cause the entire pipeline to fail, improving reliability and robustness.