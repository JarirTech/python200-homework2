

# JarirTech
# Project 07 - World Happiness Agent

from pathlib import Path
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dotenv import load_dotenv

from smolagents import CodeAgent, OpenAIServerModel, tool



# .env setup
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


DATA_PATH = Path("assignments_01/outputs/merged_happiness.csv")
FALLBACK_DIR = Path("../../python-200/assignments/resources/happiness_project")
OUTPUT_DIR = Path("assignments_07/outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = None


def clean_columns(dataframe):
    """Clean column names for easier use."""

    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return dataframe


@tool
def load_happiness_data() -> dict:
    """
    Load the World Happiness dataset.

    The function first tries the merged CSV file.
    If it is not available, it loads the yearly CSV files
    from the fallback directory and combines them.

    Returns:
        dict: Dataset shape and column names.
    """

    global df

    # --------------------------------------------------------
    # Primary dataset
   

    if DATA_PATH.exists():

        df = pd.read_csv(DATA_PATH)

    # --------------------------------------------------------
    # fallback dir
  
    else:

        all_df = []

        if not FALLBACK_DIR.exists():
            return {
                "error": (
                    f"Neither the primary dataset nor the "
                    f"fallback directory was found."
                )
            }

        for file_path in FALLBACK_DIR.glob("*.csv"):

            try:
                year = int(
                    file_path.stem.split("_")[-1]
                )
            except ValueError:
                continue

            temp_df = pd.read_csv(
                file_path,
                sep=";",
                decimal=","
            )

            # Renaming yearly dataset columns
            if "Ladder score" in temp_df.columns:
                temp_df = temp_df.rename(
                    columns={
                        "Ladder score": "Happiness score"
                    }
                )

            if "Country or region" in temp_df.columns:
                temp_df = temp_df.rename(
                    columns={
                        "Country or region": "Country name"
                    }
                )

            temp_df["year"] = year

            all_df.append(temp_df)

        if not all_df:
            return {
                "error": "No yearly happiness CSV files were found."
            }

        df = pd.concat(
            all_df,
            ignore_index=True
        )

    # --------------------------------------------------------
    # Clean columns
  

    df = clean_columns(df)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }
# ============================================================================
# Task 1 - Tool 2


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for one column.

    Args:
        column: Name of the column to summarize.

    Returns:
        dict: Descriptive statistics for the column.
    """
    global df

    if df is None:
        load_happiness_data()

    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    return df[column].describe().to_dict()


# =========================
# Task 1: Tool 3


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two numeric columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        dict: Column names, Pearson correlation, and p-value.
    """
    global df

    if df is None:
        load_happiness_data()

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns were not found."}

    try:
        data = df[[col1, col2]].dropna()

        r, p = pearsonr(data[col1], data[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4)
        }

    except Exception as e:
        return {"error": str(e)}


# ==================================================================
# Task 1: Tool 4


@tool
def get_top_n_countries(
    column: str,
    year: int,
    n: int = 5
) -> dict:
    """Return the top N countries for a column in a specific year.

    Args:
        column: Column used to rank the countries.
        year: Year to filter the dataset.
        n: Number of countries to return.

    Returns:
        dict: Top countries and their values.
    """
    global df

    if df is None:
        load_happiness_data()

    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    try:
        filtered = df[df["year"] == year]

        top = (
            filtered
            .sort_values(column, ascending=False)
            .head(n)
        )

        results = top[["country", column]].to_dict(
            orient="records"
        )

        return {
            "year": year,
            "results": results
        }

    except Exception as e:
        return {"error": str(e)}


# ========================================================================
# Task 2: Build the Agent


api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)


SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

Use the available tools for:
- loading data
- summarizing columns
- computing correlations
- ranking countries

The load_happiness_data tool loads the real World Happiness
dataset. It first tries the merged CSV and uses the required
yearly-file fallback if the merged file is unavailable.

For custom plots:
- use the real World Happiness data
- use pandas and matplotlib
- do not create fake, simulated, or random data
- do not invent values
- save plots to the requested output path

If a plot requires the dataset, call load_happiness_data()
first and use the real data loaded by the project.

"""


agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "scipy.stats"
    ],
    max_steps=8
)


# ========================================================================
# Task 3 and Task 4


if __name__ == "__main__":

    os.makedirs("outputs", exist_ok=True)

    print("\n===== WORLD HAPPINESS AGENT =====")

    # -------------------------
    # Task 3: Guided Queries
    

    queries = [
        "Load the happiness data and tell me its shape and column names.",

        "Summarize the happiness_score column.",

        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",

        "Show me the top 5 happiest countries in 2020.",

        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png."
    ]

    for query in queries:
        print("\n--- Query:", query, "---")

        response = agent.run(
            query,
            reset=False
        )

        print(response)

    # ---------------------------------------------------------------------------------
    # Task 4: My Own Questions
 
    # My query 1
    my_query_1 = (
        "What is the correlation between "
        "freedom_to_make_life_choices and happiness_score?"
    )

    response_1 = agent.run(
        my_query_1,
        reset=False
    )

    print("\n--- My Query 1 ---")
    print(response_1)


# The agent used the compute_correlation tool to answer

# this question. It calculated the correlation and returned

# the result without needing to generate any additional code.



    # My query 2
    my_query_2 = (
        "Create a histogram of happiness_score and "
        "save it as outputs/happiness_histogram.png."
    )

    response_2 = agent.run(
        my_query_2,
        reset=False
    )

    print("\n--- My Query 2 ---")
    print(response_2)

# This question required the agent to generate Python code.

# Since there was no tool available for creating charts,

# the agent attempted to use matplotlib to build and save

# the histogram.


# ====================================================================
# Task 5: Reflection

# --- Reflection ---
#
# 1. In Query 3, the agent used the p-value to decide if

# the correlation was statistically significant. Since

# the p-value was below 0.05, it considered the result

# significant.

#

# 2. I was surprised by how many different approaches the

# agent tried when creating the plot. Even when it could

# not access the data directly, it kept trying to solve

# the problem.

#

#3. One useful additional tool would be a plotting tool.

# It could create charts directly from the dataset and

# help answer visual questions about trends over time.
