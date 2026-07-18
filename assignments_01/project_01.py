####   Part 2: Mini-Project: World Happiness Pipeline


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prefect import task, flow, get_run_logger
import os



# Task 1: Load Multiple Years of Data

@task(retries=3, retry_delay_seconds=2)
def load_data():
    logger = get_run_logger()

    data_path = "csv"
    data_files = [f"{data_path}/world_happiness_{year}.csv" for year in range(2015, 2025)]

    all_df = []

    for file in data_files:
        year = int(file.split("_")[-1].split(".")[0])

        df = pd.read_csv(file, sep=";", decimal=",")

        # Standardize column names
        if "Ladder score" in df.columns:
            df["Happiness score"] = df["Ladder score"]

        if "Country or region" in df.columns:
            df.rename(columns={"Country or region": "Country name"}, inplace=True)

        df["year"] = year
        all_df.append(df)

    merged_df = pd.concat(all_df, ignore_index=True)

    os.makedirs("outputs", exist_ok=True)
    merged_df.to_csv("outputs/merged_happiness.csv", index=False)

    logger.info("Data loaded and merged successfully")

    return merged_df


# --------------------------------------------------------------------------
# Task 2: Descriptive Statistics

@task
def descriptive_stats(df):
    logger = get_run_logger()

    df_clean = df["Happiness score"].dropna()

    logger.info(f"Mean happiness: {df_clean.mean():.3f}")
    logger.info(f"Median happiness: {df_clean.median():.3f}")
    logger.info(f"Std happiness: {df_clean.std():.3f}")

    grouped_by_year = df.groupby("year")["Happiness score"].mean()
    logger.info(f"Mean by year:\n{grouped_by_year}")

    grouped_by_region = df.groupby("Regional indicator")["Happiness score"].mean()
    logger.info(f"Mean by region:\n{grouped_by_region}")

    return grouped_by_region


# ------------------------------------------------------------------------------
# Task 3: Visual Exploration

@task
def create_plots(df):
    logger = get_run_logger()
    output_path = "outputs/"

    # Histogram
    plt.figure()
    plt.hist(df["Happiness score"].dropna())
    plt.title("Happiness Distribution")
    plt.savefig(output_path + "happiness_histogram.png")
    plt.close()

    # Boxplot
    plt.figure()
    df.dropna(subset=["Happiness score"]).boxplot(column="Happiness score", by="year")
    plt.suptitle("")
    plt.title("Happiness Score by Year")
    plt.savefig(output_path + "happiness_by_year.png")
    plt.close()

    # Scatter plot
    plt.figure()
    clean_df = df[["GDP per capita", "Happiness score"]].dropna()
    plt.scatter(clean_df["GDP per capita"], clean_df["Happiness score"])
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness score")
    plt.title("GDP vs Happiness")
    plt.savefig(output_path + "gdp_vs_happiness.png")
    plt.close()

    # Heatmap
    plt.figure()
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.savefig(output_path + "correlation_heatmap.png")
    plt.close()

    logger.info("All plots saved")


# ----------------------------------------------------------------------------------------
# Task 4: Hypothesis Testing

@task
def hypothesis_tests(df):
    from scipy.stats import ttest_ind
    logger = get_run_logger()

    df_2019 = df[df["year"] == 2019]["Happiness score"]
    df_2020 = df[df["year"] == 2020]["Happiness score"]

    t_stat, p_val = ttest_ind(df_2019, df_2020, nan_policy='omit')

    if p_val < 0.05:
        conclusion = "Significant change after 2020"
    else:
        conclusion = "No significant change after 2020"

    logger.info(f"COVID test: t={t_stat:.3f}, p={p_val:.3f}")
    logger.info(conclusion)

    # Region comparison
    europe = df[df["Regional indicator"] == "Western Europe"]["Happiness score"]
    africa = df[df["Regional indicator"] == "Sub-Saharan Africa"]["Happiness score"]

    t_stat2, p_val2 = ttest_ind(europe, africa, nan_policy='omit')

    logger.info(f"Europe vs Africa: t={t_stat2:.3f}, p={p_val2:.3e}")

    return {
        "covid_test": (t_stat, p_val, conclusion),
        "region_test": (t_stat2, p_val2)
    }


# ------------------------------------------------------------------
# Task 5: Correlation and Multiple Comparisons

@task
def correlation_analysis(df):
    from scipy.stats import pearsonr
    logger = get_run_logger()

    target = "Happiness score"

    # Remove problematic columns
    EXCLUDED_COLS = ["year", "Ranking", "Happiness score", "Ladder score"]

    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col not in EXCLUDED_COLS
    ]

    alpha = 0.05
    corrected_alpha = alpha / len(numeric_cols)

    results = []

    for col in numeric_cols:
        temp_df = df[[target, col]].dropna()

        if len(temp_df) < 10:
            continue

        r, p = pearsonr(temp_df[target], temp_df[col])

        significant = p < alpha
        corrected = p < corrected_alpha
        practical = abs(r) >= 0.2

        results.append((col, r, p, significant, corrected, practical))

        logger.info(
            f"{col}: r={r:.3f}, p={p:.3e}, "
            f"sig={significant}, corrected={corrected}, practical={practical}"
        )

    return results


# ----------------------------------------------------------------------------
# Task 6: Summary Report

@task
def summary_report(df, correlations, tests):
    logger = get_run_logger()

    # Handle column inconsistency safely
    country_col = "Country name" if "Country name" in df.columns else df.columns[0]

    total_countries = df[country_col].nunique()
    total_years = df["year"].nunique()

    logger.info(f"Total countries: {total_countries}")
    logger.info(f"Total years: {total_years}")

    region_means = df.groupby("Regional indicator")["Happiness score"].mean()

    logger.info(f"Top 3 regions:\n{region_means.sort_values(ascending=False).head(3)}")
    logger.info(f"Bottom 3 regions:\n{region_means.sort_values().head(3)}")

    logger.info(f"Summary: {tests['covid_test'][2]}")

    # Filter meaningful correlations
    valid_corrs = [c for c in correlations if c[5]]

    if valid_corrs:
        best = max(valid_corrs, key=lambda x: abs(x[1]))
        logger.info(f"Strongest meaningful correlation: {best}")
    else:
        logger.info("No meaningful correlations found.")

    

# ------------------------------------------------------------
# Flow

@flow
def happiness_pipeline():
    logger = get_run_logger()

    df = load_data()
    descriptive_stats(df)
    create_plots(df)
    tests = hypothesis_tests(df)
    correlations = correlation_analysis(df)
    summary_report(df, correlations, tests)

    logger.info("Pipeline completed successfully")


# -------------------------------
# Run

if __name__ == "__main__":
    happiness_pipeline()
