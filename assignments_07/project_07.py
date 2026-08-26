# JarirTech
# Project 07 - World Happiness Agent

from pathlib import Path
import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")

from scipy.stats import pearsonr
from dotenv import load_dotenv

from smolagents import (
    CodeAgent,
    OpenAIServerModel,
    tool,
)


# ========================================================================
# .env SETUP


if load_dotenv():
    print("API key loaded successfully.")
else:
    print(
        "Warning: could not load API key. "
        "Check your .env file."
    )


# =================================================================
# PATHS




DATA_PATH = Path(
    "assignments_01/outputs/merged_happiness.csv"
)

FALLBACK_DIR = Path(
    "assignments/resources/happiness_project"
)

OUTPUT_DIR = Path(
    "assignments_07/outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Shared global DataFrame required by the assignment.
df = None


# ============================================================
# HELPER FUNCTION


def clean_columns(dataframe):
    """
    Clean and standardize DataFrame column names.

    Args:
        dataframe: Pandas DataFrame whose columns should
            be cleaned.

    Returns:
        pandas.DataFrame: DataFrame with lowercase column
            names and spaces replaced with underscores.
    """

    dataframe.columns = (
        dataframe
        .columns
        .str.strip()
        .str.lower()
        .str.replace(
            " ",
            "_",
            regex=False
        )
    )

    # Making common country column names consistent.
    if "country_name" in dataframe.columns:
        dataframe = dataframe.rename(
            columns={
                "country_name": "country"
            }
        )

    if "country_or_region" in dataframe.columns:
        dataframe = dataframe.rename(
            columns={
                "country_or_region": "country"
            }
        )

    # Making the region name easier to use for plotting.
    if "regional_indicator" in dataframe.columns:
        dataframe = dataframe.rename(
            columns={
                "regional_indicator": "region"
            }
        )

    return dataframe


# ============================================================
# TASK 1 - TOOL 1


@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    The function first tries to load the merged CSV from
    assignments_01/outputs/merged_happiness.csv. If that file
    is not available, it loads and combines the yearly CSV
    files from assignments/resources/happiness_project/.
    The final DataFrame is stored in the global df variable.

    Returns:
        A dictionary containing the dataset "shape" and
        "columns", or an error dictionary if loading fails.
    """

    global df

    # --------------------------------------------------------
    # Primary merged dataset
   

    if DATA_PATH.exists():

        try:
            df = pd.read_csv(
                DATA_PATH
            )

        except Exception as error:

            df = None

            return {
                "error": (
                    "Could not read the merged dataset: "
                    f"{error}"
                )
            }

    # --------------------------------------------------------
    # fallback
   

    else:

        if not FALLBACK_DIR.exists():

            df = None

            return {
                "error": (
                    "Neither the merged dataset nor the "
                    "required fallback directory was found."
                )
            }

        yearly_frames = []

        csv_files = sorted(
            FALLBACK_DIR.glob(
                "*.csv"
            )
        )

        if not csv_files:

            df = None

            return {
                "error": (
                    "No yearly happiness CSV files were found "
                    "in the fallback directory."
                )
            }

        for file_path in csv_files:

            # Getting the year from names such as:
            # 2015.csv
            # happiness_2015.csv

            try:
                year = int(
                    file_path
                    .stem
                    .split("_")[-1]
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

                df = None

                return {
                    "error": (
                        f"Could not read {file_path.name}: "
                        f"{error}"
                    )
                }

            # Standardize yearly dataset columns.

            if "Ladder score" in temp_df.columns:
                temp_df = temp_df.rename(
                    columns={
                        "Ladder score":
                            "Happiness score"
                    }
                )

            if "Country or region" in temp_df.columns:
                temp_df = temp_df.rename(
                    columns={
                        "Country or region":
                            "Country"
                    }
                )

            if "Country name" in temp_df.columns:
                temp_df = temp_df.rename(
                    columns={
                        "Country name":
                            "Country"
                    }
                )

            temp_df["year"] = year

            yearly_frames.append(
                temp_df
            )

        if not yearly_frames:

            df = None

            return {
                "error": (
                    "No usable yearly happiness CSV "
                    "files were found."
                )
            }

        df = pd.concat(
            yearly_frames,
            ignore_index=True
        )

    # --------------------------------------------------------
    # Clean columns
   

    df = clean_columns(
        df
    )

    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }


# ============================================================
# TASK 1 - TOOL 2


@tool
def summarize_column(
    column: str
) -> dict:
    """Return descriptive statistics for one column.

    Args:
        column: Name of the column to summarize.

    Returns:
        A dictionary containing statistics from
        pandas.Series.describe(), or an error dictionary if
        the data is not loaded or the column is not found.
    """

    global df

    if df is None:
        return {
            "error": (
                "Data is not loaded. "
                "Call load_happiness_data first."
            )
        }

    if column not in df.columns:
        return {
            "error": (
                f"Column '{column}' was not found."
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

            if hasattr(
                value,
                "item"
            ):
                value = value.item()

            cleaned_summary[key] = value

        return cleaned_summary

    except Exception as error:

        return {
            "error": str(error)
        }


# ============================================================
# TASK 1 - TOOL 3


@tool
def compute_correlation(
    col1: str,
    col2: str
) -> dict:
    """Compute the Pearson correlation between two numeric columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dictionary containing "col1", "col2", "pearson_r",
        and "p_value", or an error dictionary for invalid input.
    """

    global df

    if df is None:
        return {
            "error": (
                "Data is not loaded. "
                "Call load_happiness_data first."
            )
        }

    if col1 not in df.columns:
        return {
            "error": (
                f"Column '{col1}' was not found."
            )
        }

    if col2 not in df.columns:
        return {
            "error": (
                f"Column '{col2}' was not found."
            )
        }

    try:

        data = (
            df[
                [col1, col2]
            ]
            .dropna()
        )

        if len(data) < 2:
            return {
                "error": (
                    "Not enough valid data points "
                    "to compute the correlation."
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
                "Could not compute correlation: "
                f"{error}"
            )
        }


# ============================================================
# TASK 1 - TOOL 4

@tool
def get_top_n_countries(
    column: str,
    year: int,
    n: int = 5
) -> list[dict] | dict:
    """Return the top N countries for a column in a specific year.

    Args:
        column: Name of the column used to rank countries.
        year: Year used to filter the dataset.
        n: Number of top countries to return.

    Returns:
        A list of dictionaries containing "country" and the
        requested column value. Returns an error dictionary
        when the input is invalid.
    """

    global df

    if df is None:
        return {
            "error": (
                "Data is not loaded. "
                "Call load_happiness_data first."
            )
        }

    if column not in df.columns:
        return {
            "error": (
                f"Column '{column}' was not found."
            )
        }

    if "country" not in df.columns:
        return {
            "error": (
                "Column 'country' was not found."
            )
        }

    if "year" not in df.columns:
        return {
            "error": (
                "Column 'year' was not found."
            )
        }

    if n <= 0:
        return {
            "error": (
                "n must be greater than 0."
            )
        }

    try:

        filtered = (
            df[
                df["year"] == year
            ]
            .dropna(
                subset=[column]
            )
        )

        if filtered.empty:
            return {
                "error": (
                    f"No data was found for year {year}."
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

        return (
            top[
                ["country", column]
            ]
            .to_dict(
                orient="records"
            )
        )

    except Exception as error:

        return {
            "error": str(error)
        }


# ============================================================
# TASK 2 - BUILD THE AGENT


api_key = os.getenv(
    "OPENAI_API_KEY"
)

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not found. "
        "Check your .env file."
    )


model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)


SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

Use the available tools for loading data, summarizing columns,
computing correlations, and ranking countries.

Write Python code directly only when the tools are not sufficient,
for example when creating custom plots.

Rules:

1. Use load_happiness_data first when the dataset is not loaded.

2. load_happiness_data returns a dictionary with "shape" and
   "columns". Use that returned dictionary when answering the
   loading question.

3. Use summarize_column for column summaries.

4. Use compute_correlation for correlations.

5. For statistical significance, use p < 0.05.

6. Use get_top_n_countries for country rankings.

7. Do not create fake, simulated, or random World Happiness data.

8. For custom plots, use the real DataFrame passed as df.

9. Use pandas and matplotlib for custom plots.

10. When the user gives a plot path, save the plot to exactly
    that path. Do not change the directory or filename.

11. The project output directory is:
    assignments_07/outputs/

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
        "matplotlib.pyplot",
        "scipy.stats"
    ],
    max_steps=8
)


# ======================================================================
# TASK 3 AND TASK 4


if __name__ == "__main__":

    print(
        "\n===== WORLD HAPPINESS AGENT ====="
    )

    # ===========================================================
    # TASK 3 - GUIDED QUERIES
   

    queries = [

        # Query 1
        (
            "Load the happiness data and tell me "
            "its shape and column names."
        ),

        # Query 2
        (
            "Summarize the happiness_score column."
        ),

        # Query 3
        (
            "What is the correlation between "
            "gdp_per_capita and happiness_score? "
            "Is it statistically significant?"
        ),

        # Query 4
        (
            "Show me the top 5 happiest countries "
            "in 2020."
        ),

        # Query 5
        (
            "Plot happiness_score over the years as a line chart, "
            "with one line per region. Save the plot to "
            "assignments_07/outputs/happiness_by_region.png. "
            "Use exactly this path and do not change the "
            "directory or filename."
        )
    ]


    for query_number, query in enumerate(
        queries,
        start=1
    ):

        print(
            f"\n--- Query: {query} ---"
        )

        # Query 1 loads the shared global DataFrame.
        if query_number == 1:

            response = agent.run(
                query,
                reset=False
            )

        # After Query 1, giving the CodeAgent access to df.
        else:

            response = agent.run(
                query,
                additional_args={
                    "df": df
                },
                reset=False
            )

        print(response)


    # ========================================================
    # VERIFY QUERY 5 PLOT
 

    plot_path = Path(
        "assignments_07/outputs/happiness_by_region.png"
    )

    if plot_path.exists():

        print(
            "Verified: "
            "assignments_07/outputs/happiness_by_region.png "
            "was created by the agent."
        )

    else:

        print(
            "Plot verification failed: "
            "assignments_07/outputs/happiness_by_region.png "
            "was not created."
        )


    # ========================================================
    # TASK 4 - MY OWN QUESTIONS
   

    # --------------------------------------------------------
    # My Query 1
    # --------------------------------------------------------

    my_query_1 = (
        "What is the correlation between "
        "freedom_to_make_life_choices "
        "and happiness_score?"
    )

    response_1 = agent.run(
        my_query_1,
        additional_args={
            "df": df
        },
        reset=False
    )

    print(
        "\n--- My Query 1 ---"
    )

    print(response_1)

    # Comment:
    # This triggered tool use because the agent could use
    # compute_correlation. It did not need custom code.


    # --------------------------------------------------------
    # My Query 2
    # --------------------------------------------------------

    my_query_2 = (
        "Create a histogram of happiness_score and save it to "
        "assignments_07/outputs/happiness_histogram.png. "
        "Use exactly this path and do not change the "
        "directory or filename."
    )

    response_2 = agent.run(
        my_query_2,
        additional_args={
            "df": df
        },
        reset=False
    )

    print(
        "\n--- My Query 2 ---"
    )

    print(response_2)

    # Comment:
    # This triggered code generation because none of the
    # available tools creates a histogram. The CodeAgent had
    # to write matplotlib code using the real DataFrame.


    # ========================================================
    

    histogram_path = Path(
        "assignments_07/outputs/happiness_histogram.png"
    )

    if histogram_path.exists():

        print(
            "Verified: "
            "assignments_07/outputs/happiness_histogram.png "
            "was created by the agent."
        )

    else:

        print(
            "Histogram verification failed: "
            "assignments_07/outputs/happiness_histogram.png "
            "was not created."
        )

# ============================================================
# TASK 5 - REFLECTION

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the
#    correlation was statistically significant? Did it use
#    the p-value correctly? What threshold did it apply?
#
#    The compute_correlation tool returned Pearson r and the
#    p-value. The agent used the p-value to explain statistical
#    significance. If p < 0.05, the result was considered
#    statistically significant. The threshold was 0.05.
#
#
# 2. Did any of the agent's responses surprise you — either
#    by being more capable than you expected, or less?
#    Describe one specific example.
#
#    I was surprised that the agent generated matplotlib code
#    and created both requested plot files.
#
# 3. What one additional tool would make this agent meaningfully
#    more useful? Describe what it would do and what kind of
#    question it would help the agent answer.
#
#    A plotting tool would make the agent more useful. It could
#    create common charts such as line charts, histograms, and
#    scatter plots to answer questions about trends,
#    distributions, and comparisons.