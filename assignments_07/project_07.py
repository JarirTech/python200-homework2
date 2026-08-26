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


# ============================================================
# .env setup


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# ============================================================
# Paths
# ============================================================

DATA_PATH = Path("../assignments_01/outputs/merged_happiness.csv")

# Required fallback location
FALLBACK_DIR = Path("../../python-200/assignments/resources/happiness_project")

OUTPUT_DIR = Path("assignments_07/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = None


# ============================================================
# Helper function
# ============================================================

def clean_columns(dataframe):
    """Clean column names for easier use."""

    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return dataframe


# ============================================================
# Task 1 - Tool 1
# ============================================================

@tool
def load_happiness_data() -> dict:
    """
    Load the World Happiness dataset.

    First, this function tries to load the merged happiness CSV.
    If that file does not exist, it loads the yearly CSV files
    from assignments/resources/happiness_project/ and combines them.

    Returns:
        dict: Dataset shape and column names.
    """

    global df

    # --------------------------------------------------------
    # Try the primary merged dataset
    # --------------------------------------------------------

    if DATA_PATH.exists():

        df = pd.read_csv(DATA_PATH)

    # --------------------------------------------------------
    # Required fallback
    # --------------------------------------------------------

    else:

        all_df = []

        if not FALLBACK_DIR.exists():
            return {
                "error": (
                    "Neither the merged happiness dataset nor the "
                    "fallback directory was found."
                )
            }

        for file_path in FALLBACK_DIR.glob("*.csv"):

            try:
                year = int(file_path.stem.split("_")[-1])
            except ValueError:
                continue

            try:
                temp_df = pd.read_csv(
                    file_path,
                    sep=";",
                    decimal=","
                )
            except Exception as e:
                return {
                    "error": f"Could not read {file_path.name}: {e}"
                }

            # Rename columns used by yearly datasets
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
    # Clean column names
    # --------------------------------------------------------

    df = clean_columns(df)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }


# ============================================================
# Task 1 - Tool 2
# ============================================================

@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive statistics for one column.

    Args:
        column: Name of the column to summarize.

    Returns:
        dict: Descriptive statistics for the column.
    """

    global df

    if df is None:
        result = load_happiness_data()

        if "error" in result:
            return result

    if column not in df.columns:
        return {
            "error": f"Column '{column}' not found."
        }

    return df[column].describe().to_dict()


# ============================================================
# Task 1 - Tool 3
# ============================================================

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation between two numeric columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        dict: Column names, Pearson correlation, and p-value.
    """

    global df

    if df is None:
        result = load_happiness_data()

        if "error" in result:
            return result

    if col1 not in df.columns:
        return {
            "error": f"Column '{col1}' not found."
        }

    if col2 not in df.columns:
        return {
            "error": f"Column '{col2}' not found."
        }

    try:
        data = df[[col1, col2]].dropna()

        r, p = pearsonr(
            data[col1],
            data[col2]
        )

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
            "significant": bool(p < 0.05)
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# ============================================================
# Task 1 - Tool 4
# ============================================================

@tool
def get_top_n_countries(
    column: str,
    year: int,
    n: int = 5
) -> list:
    """
    Return the top N countries for a column in a specific year.

    Args:
        column: Column used to rank the countries.
        year: Year to filter the dataset.
        n: Number of countries to return.

    Returns:
        list: List of dictionaries containing country names
        and their requested values.
    """

    global df

    if df is None:
        result = load_happiness_data()

        if "error" in result:
            return [result]

    if column not in df.columns:
        return [
            {"error": f"Column '{column}' not found."}
        ]

    if "country" not in df.columns:
        return [
            {"error": "Column 'country' not found."}
        ]

    try:
        filtered = df[df["year"] == year]

        top = (
            filtered
            .sort_values(
                column,
                ascending=False
            )
            .head(n)
        )

        return top[
            ["country", column]
        ].to_dict(
            orient="records"
        )

    except Exception as e:
        return [
            {"error": str(e)}
        ]


# ============================================================
# Task 2 - Build the Agent
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=api_key
)


SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

Use the available tools for:
- loading data
- summarizing columns
- computing correlations
- ranking countries


The dataset is stored in a global pandas DataFrame named `df`

after load_happiness_data() is called.

When writing custom code for plots, use the existing global
DataFrame `df` directly instead of calling load_happiness_data()
and attempting to convert its return value into a DataFrame.
For custom plots:
- use the real World Happiness data
- use pandas and matplotlib
- do not create fake, simulated, or random data
- do not invent values
- save plots to the requested output path



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
        "numpy",
        "matplotlib",
        "matplotlib.pyplot",
        "scipy",
        "scipy.stats"
    ],
    max_steps=8
)


# ============================================================
# Task 3 and Task 4
# ============================================================

if __name__ == "__main__":

    print("\n===== WORLD HAPPINESS AGENT =====")

    # ========================================================
    # Task 3 - Guided Queries
    # ========================================================

    queries = [
        "Load the happiness data and tell me its shape and column names.",

        "Summarize the happiness_score column.",

        (
            "What is the correlation between gdp_per_capita "
            "and happiness_score? Is it statistically significant?"
        ),

        "Show me the top 5 happiest countries in 2020.",

        (
            "Plot happiness_score over the years as a line chart, "
            "with one line per region. Save the plot to "
            "outputs/happiness_by_region.png."
        )
    ]

    for query in queries:

        print("\n" + "=" * 60)
        print("Q:", query)
        print("=" * 60)

        response = agent.run(
            query,
            reset=False
        )

        print(response)

    # ========================================================
    # Task 4 - My Own Questions
    # ========================================================

    # --------------------------------------------------------
    # My Query 1
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # My Query 2
    # --------------------------------------------------------

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

    # ========================================================
    # Task 4 Reflection
    # ========================================================

    # The first question used the compute_correlation tool
    # because the tool could calculate the Pearson correlation.
    #
    # The second question required the agent to create a
    # histogram using the real happiness data and matplotlib.


# ============================================================
# Task 5 - Reflection
# ============================================================

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the
#    correlation was statistically significant? Did it use
#    the p-value correctly? What threshold did it apply?
#
#    The agent used the p-value. It considered the correlation
#    significant when p < 0.05.
#
# 2. Did any of the agent's responses surprise you — either
#    by being more capable than you expected, or less?
#    Describe one specific example.
#
#    I was surprised that the agent correctly reasoned about
#    how to create the plot, but it could not access the global
#    DataFrame used by the tools. As a result, the plotting
#    query failed even though the analytical tool queries worked.#
# 3. What one additional tool would make this agent meaningfully
#    more useful? Describe what it would do and what kind of
#    question it would help the agent answer.
#
#    A plotting tool would make the agent more useful.
#    It could create different charts to answer questions
#    about trends and comparisons.