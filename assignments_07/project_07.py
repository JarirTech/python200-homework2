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
# ============================================================

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# assignments_07/
#     project_07.py
#     outputs/
#
# assignments_01/
#     outputs/
#         merged_happiness.csv
#
# resources/
#     happiness_project/
#         yearly happiness CSV files

DATA_PATH = (
    PROJECT_DIR
    / ".."
    / "assignments_01"
    / "outputs"
    / "merged_happiness.csv"
).resolve()


# Required fallback location:
# assignments/resources/happiness_project/
# when project_07.py is inside assignments_07/

FALLBACK_DIR = (
    PROJECT_DIR
    / ".."
    / "resources"
    / "happiness_project"
).resolve()


OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Global DataFrame used by the tools.
df = None


# ============================================================
# Helper function
# ============================================================

def clean_columns(dataframe):
    """
    Clean column names for easier use.

    Converts column names to lowercase, strips leading and
    trailing spaces, and replaces spaces with underscores.

    Args:
        dataframe: Pandas DataFrame whose column names should
            be cleaned.

    Returns:
        pandas.DataFrame: DataFrame with cleaned column names.
    """

    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Make common country column names consistent.
    if "country_name" in dataframe.columns:
        dataframe = dataframe.rename(
            columns={"country_name": "country"}
        )

    # Handle another common World Happiness column name.
    if "country_or_region" in dataframe.columns:
        dataframe = dataframe.rename(
            columns={"country_or_region": "country"}
        )

    return dataframe


# ============================================================
# Task 1 - Tool 1
# ============================================================

@tool
def load_happiness_data() -> dict:
    """
    Load the World Happiness dataset.

    The function first attempts to load the merged happiness
    CSV from assignments_01/outputs/. If that file does not
    exist, it loads the yearly CSV files from the required
    resources/happiness_project/ fallback directory and
    combines them into one DataFrame.

    Args:
        None.

    Returns:
        dict: A dictionary containing the dataset shape and
        column names, or an error message if the dataset
        cannot be loaded.
    """

    global df

    # --------------------------------------------------------
    # Try the primary merged dataset first.
    # --------------------------------------------------------

    if DATA_PATH.exists():

        try:
            df = pd.read_csv(DATA_PATH)

        except Exception as error:
            return {
                "error": (
                    f"Could not read merged dataset: {error}"
                )
            }

    # --------------------------------------------------------
    # Required fallback.
    # --------------------------------------------------------

    else:

        all_df = []

        if not FALLBACK_DIR.exists():
            return {
                "error": (
                    "Neither the merged happiness dataset nor "
                    "the fallback directory was found.\n"
                    f"Primary path: {DATA_PATH}\n"
                    f"Fallback path: {FALLBACK_DIR}"
                )
            }

        csv_files = sorted(FALLBACK_DIR.glob("*.csv"))

        if not csv_files:
            return {
                "error": (
                    "No yearly happiness CSV files were found "
                    f"in {FALLBACK_DIR}."
                )
            }

        for file_path in csv_files:

            # Extract year from filename.
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

            except Exception as error:
                return {
                    "error": (
                        f"Could not read {file_path.name}: "
                        f"{error}"
                    )
                }

            # Rename columns used by yearly datasets.
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
                "error": (
                    "No usable yearly happiness CSV files "
                    "were found."
                )
            }

        df = pd.concat(
            all_df,
            ignore_index=True
        )

    # --------------------------------------------------------
    # Clean column names.
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
        dict: Descriptive statistics for the requested column,
        or an error message if the column is not available.
    """

    global df

    # Load data automatically if necessary.
    if df is None:

        result = load_happiness_data()

        if "error" in result:
            return result

    if column not in df.columns:
        return {
            "error": (
                f"Column '{column}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )
        }

    try:

        summary = df[column].describe().to_dict()

        # Convert values to standard Python types where possible.
        cleaned_summary = {}

        for key, value in summary.items():

            if hasattr(value, "item"):
                value = value.item()

            cleaned_summary[key] = value

        return cleaned_summary

    except Exception as error:

        return {
            "error": str(error)
        }


# ============================================================
# Task 1 - Tool 3
# ============================================================

@tool
def compute_correlation(
    col1: str,
    col2: str
) -> dict:
    """
    Compute the Pearson correlation between two numeric columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        dict: A dictionary containing the two column names,
        Pearson correlation coefficient, p-value, and whether
        the correlation is statistically significant at the
        0.05 level.
    """

    global df

    # Load data automatically if necessary.
    if df is None:

        result = load_happiness_data()

        if "error" in result:
            return result

    if col1 not in df.columns:
        return {
            "error": (
                f"Column '{col1}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )
        }

    if col2 not in df.columns:
        return {
            "error": (
                f"Column '{col2}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )
        }

    try:

        data = df[[col1, col2]].dropna()

        if len(data) < 2:
            return {
                "error": (
                    "Not enough valid data points to "
                    "calculate correlation."
                )
            }

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

    except Exception as error:

        return {
            "error": (
                f"Could not compute correlation: {error}"
            )
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
        year: Year used to filter the dataset.
        n: Number of countries to return.

    Returns:
        list: A list of dictionaries containing country names
        and their values for the requested column, or an error
        dictionary if the requested data is unavailable.
    """

    global df

    # Load data automatically if necessary.
    if df is None:

        result = load_happiness_data()

        if "error" in result:
            return [result]

    if column not in df.columns:
        return [
            {
                "error": (
                    f"Column '{column}' not found. "
                    f"Available columns: {df.columns.tolist()}"
                )
            }
        ]

    if "country" not in df.columns:
        return [
            {
                "error": (
                    "Column 'country' not found."
                )
            }
        ]

    if "year" not in df.columns:
        return [
            {
                "error": (
                    "Column 'year' not found."
                )
            }
        ]

    try:

        filtered = df[df["year"] == year]

        if filtered.empty:
            return [
                {
                    "error": (
                        f"No data found for year {year}."
                    )
                }
            ]

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

    except Exception as error:

        return [
            {
                "error": str(error)
            }
        ]


# ============================================================
# Task 2 - Build the CodeAgent
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not found. "
        "Check your .env file."
    )


model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=api_key
)


SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

Use the available tools for:
- loading the World Happiness data
- summarizing columns
- computing Pearson correlations
- ranking countries by a column and year

Important rules:

1. Use the tools when the question requires data from the dataset.

2. Do not guess values that are not present in the data.

3. If the data has not been loaded, use load_happiness_data first.

4. For correlations, report both Pearson r and the p-value.
   Consider a correlation statistically significant when p < 0.05.

5. For country rankings, use get_top_n_countries.

6. For custom plots, use the real World Happiness data.
   Do not create fake, simulated, or random data.

7. Use pandas and matplotlib for custom plots.

8. Save requested plots to the requested output path.

9. The project output directory is:
   assignments_07/outputs/

10. Keep answers concise and student-friendly.
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

        # Guided Query 1
        "Load the happiness data and tell me its shape and column names.",

        # Guided Query 2
        "Summarize the happiness_score column.",

        # Guided Query 3
        (
            "What is the correlation between "
            "gdp_per_capita and happiness_score? "
            "Is it statistically significant?"
        ),

        # Guided Query 4
        "Show me the top 5 happiest countries in 2020.",

        # Guided Query 5
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

    # --------------------------------------------------------
    # Verify the required Guided Query 5 output.
    # --------------------------------------------------------

    happiness_by_region_path = (
        OUTPUT_DIR / "happiness_by_region.png"
    )

    print("\nGuided Query 5 plot verification:")

    if happiness_by_region_path.exists():
        print(
            "Verified: happiness_by_region.png "
            "was saved to outputs/."
        )
    else:
        print(
            "Warning: happiness_by_region.png "
            "was not found in outputs/."
        )

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

    # The agent used the compute_correlation tool because
    # this question required calculating a Pearson correlation
    # from the World Happiness dataset. No custom code
    # generation was needed for this question.

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

    # The agent generated and ran custom Python code using
    # pandas and matplotlib to create the histogram from the
    # real World Happiness data. This question required
    # code generation rather than one of the four data tools.


# ============================================================
# Task 5 - Reflection
# ============================================================

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the
#    correlation was statistically significant? Did it use
#    the p-value correctly? What threshold did it apply?
#
#    The agent used the p-value to determine statistical
#    significance. It considered the correlation significant
#    when p < 0.05.
#
#
# 2. Did any of the agent's responses surprise you — either
#    by being more capable than you expected, or less?
#    Describe one specific example.
#
#    I was surprised that the agent could use the data-analysis
#    tools to answer questions about correlations and country
#    rankings. It was also interesting that the CodeAgent could
#    generate and run Python code with pandas and matplotlib for
#    custom plotting.
#
#
# 3. What one additional tool would make this agent meaningfully
#    more useful? Describe what it would do and what kind of
#    question it would help the agent answer.
#
#    A plotting tool would make this agent more useful. It could
#    create different charts from the World Happiness data,
#    helping answer questions about trends, distributions,
#    and comparisons between countries or regions.