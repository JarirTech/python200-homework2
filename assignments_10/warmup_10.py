# JarirTech

from time import sleep


# =========================================================
# Part 1: Warmup
# =========================================================


# =========================================================
# ML vs. LLM in Pipelines
# =========================================================


# ---------------- ML/LLM Question 1 ----------------

# The ML classifier produces a binary prediction, such as good for running
# or skip, and it also produces a confidence score.
# It works well because it was trained on labeled numeric weather data.
# The LLM produces a natural-language recommendation based on the weather
# features and the ML prediction.
# The LLM is better for language because it can generate clear human-readable text.
# If we swapped them, the LLM could give less consistent binary predictions,
# while the ML classifier cannot generate a natural-language recommendation.


# ---------------- ML/LLM Question 2 ----------------

# Converting a date string like "2023-07-04" to day-of-week:
# I would use deterministic code because the answer should always be the same.

# Classifying a job posting as entry-level, mid-level, or senior:
# I would use an LLM because it can understand freeform job description text.

# Predicting customer churn using 15 numeric features and labeled training data:
# I would use a trained ML model because it can learn patterns from labeled numeric data.

# Normalizing inconsistent city names:
# I would use an LLM because it can understand different ways people write the same city name.

# Summing a column of revenue figures:
# I would use deterministic code because addition is a simple exact calculation.


# ---------------- ML/LLM Question 3 ----------------

# Incremental processing means processing only records that have not already
# been processed.
# It is important because the pipeline does not need to repeat work for records
# that are already stored in weather_enriched.
# If all 365 records were processed every time, the ML model and especially
# the LLM would repeat unnecessary work.
# This would increase API cost and processing time.
# It could also change previous LLM summaries because LLM output may vary
# between runs, which could reduce data consistency.


# =========================================================
# Prompt Design
# =========================================================


# ---------------- Prompt Question 1 ----------------

# SYSTEM_PROMPT = (
#     "You are writing a two-sentence running recommendation for a daily weather app. "
#     "The first sentence must state whether the prediction is good for running or skip. "
#     "The second sentence must explain the reason using the weather conditions. "
#     "Write exactly two sentences. "
#     "Do not use bullet points, headers, or extra explanation."
# )

# To validate two sentences instead of one, I would change the validation logic
# so it checks that exactly two sentences are returned.
# I would also check that the first sentence states the prediction and the second
# sentence explains the reasoning.


# ---------------- Prompt Question 2 ----------------

def call_with_retry(client, messages, max_retries=3):
    """
    Call the OpenAI API and retry up to max_retries times
    if an exception happens.
    Wait 2 seconds between attempts.
    Return None after the final failure.
    """

    max_attempts = max_retries + 1

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )

            return response

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries:
                print("Retrying in 2 seconds...")
                sleep(2)

            else:
                print("Final attempt failed. Returning None.")
                return None


# I would use this retry function in a production pipeline when calling an API
# because temporary network or service errors can happen.
# Retrying can help the pipeline recover from short temporary failures
# instead of failing immediately.