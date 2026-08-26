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

# Primary merged dataset:
# assignments_01/
#     outputs/
#         merged_happiness.csv
#
# Project:
# assignments_07/
#     project_07.py
#     outputs/

DATA_PATH = (
    PROJECT_DIR
    / ".."
    / "assignments_01"
    / "outputs"
    / "merged_happiness.csv"
).resolve()


# Required fallback location inside the repository:
# assignments/
#     resources/
#         happiness_project/

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
    Clean and standardize column names in a DataFrame.

    Args:
        dataframe: The pandas DataFrame whose column names
            should be cleaned.

    Returns:
        pandas.DataFrame: The DataFrame with lowercase column
            names, stripped whitespace, and spaces replaced
            with underscores.
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

    if "country_or_region" in dataframe.columns:
        dataframe = dataframe.rename(
            columns={"country_or_region": "country"}
        )

    # Make common region column names consistent.
    if "regional_indicator" in dataframe.columns:
        dataframe = dataframe.rename(
            columns={"regional_indicator": "region"}
        )

    if "region_name" in dataframe.columns:
        dataframe = dataframe.rename(
            columns={"region_name": "region"}
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
    dataset from assignments_01/outputs/merged_happiness.csv.
    If that file does not exist, the function loads the yearly
    CSV files from assignments/resources/happiness_project/
    and combines them into one DataFrame.

    Returns:
        dict: A dictionary containing the dataset shape and
        column names, or an error message if the data cannot
        be loaded.
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
                    "the fallback directory was found."
                )
            }

        csv_files = sorted(
            FALLBACK_DIR.glob("*.csv")
        )

        if not csv_files:
            return {
                "error": (
                    "No yearly happiness CSV files were found "
                    "in the fallback directory."
                )
            }

        for file_path in csv_files:

            # Extract year from filename.
            try:
                year = int(
                    file_path.stem.split("_")[-1]
                )
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
    Return descriptive statistics for one dataset column.

    Args:
        column: The name of the column to summarize.

    Returns:
        dict: A dictionary containing descriptive statistics
        for the requested column, or an error message if the
        column does not exist.
    """

    global df

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

        summary = (
            df[column]
            .describe()
            .to_dict()
        )

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
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.

    Returns:
        dict: A dictionary containing col1, col2, pearson_r,
        and p_value.
    """

    global df

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

        data = df[
            [col1, col2]
        ].dropna()

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
            "pearson_r": round(
                float(r),
                4
            ),
            "p_value": round(
                float(p),
                4
            )
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
) -> dict:
    """
    Return the top N countries for a column in a specific year.

    Args:
        column: The column used to rank the countries.
        year: The year used to filter the dataset.
        n: The number of countries to return.

    Returns:
        dict: A dictionary containing the top countries and
        their values, or an error message if the requested
        column, year, or country data is unavailable.
    """

    global df

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

    if "country" not in df.columns:
        return {
            "error": (
                "Column 'country' not found."
            )
        }

    if "year" not in df.columns:
        return {
            "error": (
                "Column 'year' not found."
            )
        }

    if n <= 0:
        return {
            "error": "n must be greater than 0."
        }

    try:

        filtered = df[
            df["year"] == year
        ]

        if filtered.empty:
            return {
                "error": (
                    f"No data found for year {year}."
                )
            }

        top = (
            filtered
            .sort_values(
                column,
                ascending=False
            )
            .head(n)
        )

        results = top[
            ["country", column]
        ].to_dict(
            orient="records"
        )

        return {
            "year": year,
            "column": column,
            "n": n,
            "results": results
        }

    except Exception as error:

        return {
            "error": str(error)
        }


# ============================================================
# Task 2 - Build the CodeAgent
# ============================================================

api_key = os.getenv(
    "OPENAI_API_KEY"
)

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

1. Use the tools when the question requires data from
   the World Happiness dataset.

2. Do not guess values that are not present in the data.

3. If the data has not been loaded, use
   load_happiness_data first.

4. For correlations, use compute_correlation and report
   Pearson r and the p-value. A p-value below 0.05 is
   considered statistically significant.

5. For country rankings, use get_top_n_countries.

6. For custom plots, use the real World Happiness data.

7. Do not create fake, simulated, or random data.

8. Use pandas and matplotlib for custom plots.

9. Save requested plots to the exact path requested by
   the user.

10. The project output directory is:
    outputs/

11. For the regional happiness line chart, use:
    - happiness_score on the y-axis
    - year on the x-axis
    - one line for each region
    - save the result as:
      outputs/happiness_by_region.png

12. Keep answers concise and student-friendly.
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
# Helper for required Query 5 plot
# ============================================================

def create_happiness_by_region_plot():
    """
    Create the required regional happiness line chart.

    Uses the loaded World Happiness DataFrame and saves the
    chart to outputs/happiness_by_region.png.

    Returns:
        bool: True if the plot was created successfully,
        otherwise False.
    """

    global df

    if df is None:
        result = load_happiness_data()

        if "error" in result:
            print(
                "Could not create required plot:",
                result
            )
            return False

    required_columns = [
        "year",
        "region",
        "happiness_score"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print(
            "Could not create required regional plot. "
            f"Missing columns: {missing}"
        )
        return False

    plot_data = (
        df[
            [
                "year",
                "region",
                "happiness_score"
            ]
        ]
        .dropna()
        .groupby(
            ["year", "region"],
            as_index=False
        )["happiness_score"]
        .mean()
    )

    if plot_data.empty:
        print(
            "Could not create required regional plot "
            "because no valid data was available."
        )
        return False

    plt.figure(
        figsize=(10, 6)
    )

    for region_name, region_data in plot_data.groupby(
        "region"
    ):

        plt.plot(
            region_data["year"],
            region_data["happiness_score"],
            marker="o",
            label=region_name
        )

    plt.title(
        "World Happiness Score by Region Over Time"
    )

    plt.xlabel("Year")
    plt.ylabel("Happiness Score")

    plt.legend(
        title="Region",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "happiness_by_region.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Required plot saved to: {output_path}"
    )

    return True


# ============================================================
# Task 3 and Task 4
# ============================================================

if __name__ == "__main__":

    print(
        "\n===== WORLD HAPPINESS AGENT ====="
    )

    print(
        "\nPrimary data path:"
    )
    print(DATA_PATH)

    print(
        "\nFallback directory:"
    )
    print(FALLBACK_DIR)

    print(
        "\nOutput directory:"
    )
    print(OUTPUT_DIR)

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

    for query_number, query in enumerate(
        queries,
        start=1
    ):

        print(
            "\n" + "=" * 60
        )

        print(
            f"Guided Query {query_number}:"
        )

        print(query)

        print(
            "=" * 60
        )

        response = agent.run(
            query,
            reset=False
        )

        print(response)

        # ----------------------------------------------------
        # Query 5 verification.
        # ----------------------------------------------------

        if query_number == 5:

            required_plot = (
                OUTPUT_DIR
                / "happiness_by_region.png"
            )

            if required_plot.exists():

                print(
                    "Verified: "
                    f"{required_plot} was created."
                )

            else:

                print(
                    "The agent did not create the required "
                    "plot file, so the required regional "
                    "plot will now be created from the "
                    "World Happiness data."
                )

                create_happiness_by_region_plot()

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

    print(
        "\n--- My Query 1 ---"
    )

    print(response_1)

    # The compute_correlation tool is used because this
    # question asks for a Pearson correlation between two
    # columns in the World Happiness dataset.


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

    print(
        "\n--- My Query 2 ---"
    )

    print(response_2)

    # The CodeAgent generates and runs Python code with
    # pandas and matplotlib because this question requires
    # creating and saving a custom histogram.


# ============================================================
# Task 5 - Reflection
# ============================================================

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the
#    correlation was statistically significant? Did it use
#    the p-value correctly? What threshold did it apply?
#
#    The agent used the p-value returned by the
#    compute_correlation tool. A correlation was considered
#    statistically significant when p < 0.05.
#
#
# 2. Did any of the agent's responses surprise you — either
#    by being more capable than you expected, or less?
#    Describe one specific example.
#
#    I was surprised that the CodeAgent could use the
#    World Happiness data and create a custom regional
#    line chart using pandas and matplotlib. The tool
#    functions handled the structured data questions,
#    while the CodeAgent could generate custom Python
#    code for the plotting task.
#
#
# 3. What one additional tool would make this agent meaningfully
#    more useful? Describe what it would do and what kind of
#    question it would help the agent answer.
#
#    A plotting tool would make the agent more useful.
#    It could create charts directly from the World
#    Happiness data, helping answer questions about trends,
#    distributions, and comparisons between countries
#    or regions.