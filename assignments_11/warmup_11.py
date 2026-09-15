
#------Part 1: Warmup----------

#----Prefect Orchestration------
#--------Prefect Question 1------

#what is the difference between a @task and a @flow in Prefect?

# A @task in prefect is a single unit of work that only do one task or one job in a serie of workflow

# a @flow is the orchestrator that act as a manager that calls tasks in order and manages the run as a whole.



# You have a helper function that converts a temperature from Celsius to Fahrenheit -- a pure, in-memory calculation 
# with no I/O. Would you decorate it with @task? Why or why not?

# I would not decorate this helper function with @task because it only does a simple temperature conversion
#  from C to F. It does not perform I/O or take a long time to run, so Prefect task tracking is unnecessary. 
# Using a normal Python function is simpler and faster for this kind of small in-memory calculation.

#==========================================================================================================

#----Prefect Question 2-----
# Write the decorator (just the decorator line, not the full function) for a task named call_api that retries up
#  to 3 times with a 30-second delay between attempts.

#@task(retries=3, retry_delay_seconds=30)
#def call_api():

#=============================================================================================================

#------Prefect Question 3--------

# You run your pipeline and the Prefect UI shows: extract is Completed, transform is Failed, load never ran.
# In a comment block, describe: where in the UI do you look to understand what went wrong, and what specific 
# information would you expect to find there?

# As long extracte was completed and transform is failed. I will double check the log for errors in the transform
#  task. I would expect to find error messages that indicate what went wrong during the transform step.

#============================================================================================================

#-----Production Patterns-----
#----------------------Production Question 1

#raise_for_status() is a built-in method used in Python HTTP libraries to automatically check if a website request failed.

# This function is better than using the if response.status_code != 200: print("error") in a pipeline task, because 
#if the API returns a 404 or 500, the pipeline continues with an empty or malformed response. That can lead to corrupted 
# downstream data, misleading results, or a pipeline that appears successful even though the data is wrong.

#===========================================================================================================

#------Production Question 2

# Your load_raw task uses upsert with on_conflict="date" instead of insert. The pipeline crashes halfway through the transform step. 
# You fix the bug and re-run from the beginning. In a comment block, explain: what does upsert protect you from in this scenario, 
# and what would happen if you had used plain insert instead?

# Upsert protects from having creating duplicates. With  on_conflict="date" an existing date will be updated instead
# of inserting another row with the same date.
# If I used plain insert, it could create duplicate rows, or it could fail if the date column
# is unique or a primary key.

#============================================================================================

#------Production Question 3----

#Write a task stub — just the function signature, decorator, and a single log line — that uses get_run_logger() 
# to log an INFO message saying how many enrichment records were upserted. The function should accept enrichment_records (a list) as its argument.


# @task
# def log_records(enrichment_records: list):
#    logger = get_run_logger()
#    logger.info(f"Records loaded: {len(records)}")

#=============================================================================================

#---------------------------Production Question 4
# In a comment block, explain how the incremental processing check in the transform task contributes to idempotency.
#  If you removed it and the pipeline ran the ML and LLM steps on all 365 records every time, what would be the practical
#  consequences (in terms of cost, time, and data correctness)?

# Incremental processing means the pipeline only works on new records
# that have not been processed yet.
# If I remove this check, the ML and LLM will process all 365 records every time.
# This will cost more money because the OpenAI API will be called again.
# It will also take more time to finish the pipeline.
# The LLM may also give different answers for the same records,
# which could change old data and make the results less consistent.