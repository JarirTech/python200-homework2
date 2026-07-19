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

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "csv")

    data_files = [
        os.path.join(data_path, f"world_happiness_{year}.csv")
        for year in range(2015, 2025)
    ]
    all_df = []

    for file in data_files:
        year = int(file.split("_")[-1].split(".")[0])

        df = pd.read_csv(file, sep=";", decimal=",")

        # Standardize column names
        # if "Ladder score" in df.columns:
        #     df["Happiness score"] = df["Ladder score"]
        if "Ladder score" in df.columns:
            df.rename(columns={"Ladder score": "Happiness score"}, inplace=True)

        
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
    #plt.title("Happiness Distribution")
    plt.title("Happiness Scores Across All Years")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")

    plt.savefig(output_path + "happiness_histogram.png")
    plt.close()
    logger.info("Saved happiness_histogram.png")

    # Boxplot
    plt.figure()
    df.dropna(subset=["Happiness score"]).boxplot(column="Happiness score", by="year")
    plt.suptitle("")
    #plt.title("Happiness Score by Year")
    plt.title("Happiness Score Distributions Across Years")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")

    plt.savefig(output_path + "happiness_by_year.png")
    plt.close()
    logger.info("Saved happiness_by_year.png")


    # Scatter plot
    plt.figure()
    clean_df = df[["GDP per capita", "Happiness score"]].dropna()
    plt.scatter(clean_df["GDP per capita"], clean_df["Happiness score"])
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness score")
    #plt.title("GDP vs Happiness")
    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")

    plt.savefig(output_path + "gdp_vs_happiness.png")
    plt.close()
    logger.info("Saved gdp_vs_happiness.png")

    # Heatmap
    plt.figure()
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True)
    plt.title("Correlation Heatmap")

    plt.savefig(output_path + "correlation_heatmap.png")
    plt.close()
    logger.info("Saved correlation_heatmap.png")
    logger.info("All plots saved")


# ----------------------------------------------------------------------------------------
# Task 4: Hypothesis Testing

# @task
# def hypothesis_tests(df):
#     from scipy.stats import ttest_ind
#     logger = get_run_logger()

#     df_2019 = df[df["year"] == 2019]["Happiness score"]
#     df_2020 = df[df["year"] == 2020]["Happiness score"]

#     t_stat, p_val = ttest_ind(df_2019, df_2020, nan_policy='omit')

#     if p_val < 0.05:
#         conclusion = "Significant change after 2020"
#     else:
#         conclusion = "No significant change after 2020"

#     logger.info(f"COVID test: t={t_stat:.3f}, p={p_val:.3f}")
#     logger.info(conclusion)

#     # Region comparison
#     europe = df[df["Regional indicator"] == "Western Europe"]["Happiness score"]
#     africa = df[df["Regional indicator"] == "Sub-Saharan Africa"]["Happiness score"]

#     t_stat2, p_val2 = ttest_ind(europe, africa, nan_policy='omit')

#     logger.info(f"Europe vs Africa: t={t_stat2:.3f}, p={p_val2:.3e}")

#     return {
#         "covid_test": (t_stat, p_val, conclusion),
#         "region_test": (t_stat2, p_val2)
#     }


@task
def hypothesis_tests(df):
    from scipy.stats import ttest_ind
    logger = get_run_logger()

    # 
    # 2019 vs 2020 Comparison
    
    df_2019 = df[df["year"] == 2019]["Happiness score"].dropna()
    df_2020 = df[df["year"] == 2020]["Happiness score"].dropna()

    mean_2019 = df_2019.mean()
    mean_2020 = df_2020.mean()

    t_stat, p_val = ttest_ind(df_2019, df_2020, nan_policy="omit")

    logger.info(f"2019 mean happiness score: {mean_2019:.3f}")
    logger.info(f"2020 mean happiness score: {mean_2020:.3f}")
    logger.info(f"2019 vs 2020 t-test: t={t_stat:.3f}, p={p_val:.3f}")

    if p_val < 0.05:
        conclusion = (
            "At alpha = 0.05, there is a statistically significant difference "
            "between the 2019 and 2020 happiness scores."
        )
    else:
        conclusion = (
            "At alpha = 0.05, there is no statistically significant difference "
            "between the 2019 and 2020 happiness scores."
        )

    logger.info(conclusion)

    # 
    # Western Europe vs Sub-Saharan Africa
    # 
    europe = df[df["Regional indicator"] == "Western Europe"]["Happiness score"].dropna()
    africa = df[df["Regional indicator"] == "Sub-Saharan Africa"]["Happiness score"].dropna()

    europe_mean = europe.mean()
    africa_mean = africa.mean()

    t_stat2, p_val2 = ttest_ind(europe, africa, nan_policy="omit")

    logger.info(f"Western Europe mean happiness score: {europe_mean:.3f}")
    logger.info(f"Sub-Saharan Africa mean happiness score: {africa_mean:.3f}")
    logger.info(f"Western Europe vs Sub-Saharan Africa: t={t_stat2:.3f}, p={p_val2:.3e}")

    if p_val2 < 0.05:
        logger.info(
            "At alpha = 0.05, there is a statistically significant difference "
            "between Western Europe and Sub-Saharan Africa."
        )
    else:
        logger.info(
            "At alpha = 0.05, there is no statistically significant difference "
            "between Western Europe and Sub-Saharan Africa."
        )

    return {
        "covid_test": (t_stat, p_val, conclusion),
        "region_test": (t_stat2, p_val2)
    }

# ------------------------------------------------------------------
# Task 5: Correlation and Multiple Comparisons

# @task
# def correlation_analysis(df):
#     from scipy.stats import pearsonr
#     logger = get_run_logger()

#     target = "Happiness score"

#     # Removing problematic columns
#     EXCLUDED_COLS = ["year", "Ranking", "Happiness score", "Ladder score"]

#     numeric_cols = [
#         col for col in df.select_dtypes(include="number").columns
#         if col not in EXCLUDED_COLS
#     ]

#     alpha = 0.05
#     corrected_alpha = alpha / len(numeric_cols)

#     results = []

#     for col in numeric_cols:
#         temp_df = df[[target, col]].dropna()

#         if len(temp_df) < 10:
#             continue

#         r, p = pearsonr(temp_df[target], temp_df[col])

#         significant = p < alpha
#         corrected = p < corrected_alpha
#         practical = abs(r) >= 0.2

#         results.append((col, r, p, significant, corrected, practical))

#         logger.info(
#             f"{col}: r={r:.3f}, p={p:.3e}, "
#             f"sig={significant}, corrected={corrected}, practical={practical}"
#         )

#     return results

@task
def correlation_analysis(df):
    from scipy.stats import pearsonr
    logger = get_run_logger()

    target = "Happiness score"

    # Removing columns that should not be tested
    EXCLUDED_COLS = ["year", "Ranking", "Happiness score", "Ladder score"]

    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col not in EXCLUDED_COLS
    ]

    alpha = 0.05
    corrected_alpha = alpha / len(numeric_cols)

    logger.info(f"Alpha: {alpha}")
    logger.info(f"Bonferroni corrected alpha: {corrected_alpha:.6f}")

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
            f"{col}: "
            f"r={r:.3f}, "
            f"p={p:.3e}, "
            f"Significant={significant}, "
            f"Bonferroni Significant={corrected}, "
            f"Practical={practical}"
        )

    # Report variables that remain significant after Bonferroni correction
    logger.info("Variables significant after Bonferroni correction:")

    corrected_results = [r for r in results if r[4]]

    if corrected_results:
        for result in corrected_results:
            logger.info(
                f"{result[0]} "
                f"(r={result[1]:.3f}, p={result[2]:.3e})"
            )
    else:
        logger.info("No variables remained significant after Bonferroni correction.")

    return results
# ----------------------------------------------------------------------------
# Task 6: Summary Report

# @task
# def summary_report(df, correlations, tests):
#     logger = get_run_logger()

#     # Handle column inconsistency safely
#     country_col = "Country name" if "Country name" in df.columns else df.columns[0]

#     total_countries = df[country_col].nunique()
#     total_years = df["year"].nunique()

#     logger.info(f"Total countries: {total_countries}")
#     logger.info(f"Total years: {total_years}")

#     region_means = df.groupby("Regional indicator")["Happiness score"].mean()

#     logger.info(f"Top 3 regions:\n{region_means.sort_values(ascending=False).head(3)}")
#     logger.info(f"Bottom 3 regions:\n{region_means.sort_values().head(3)}")

#     logger.info(f"Summary: {tests['covid_test'][2]}")

#     # Filter meaningful correlations
#     valid_corrs = [c for c in correlations if c[5]]

#     if valid_corrs:
#         best = max(valid_corrs, key=lambda x: abs(x[1]))
#         logger.info(f"Strongest meaningful correlation: {best}")
#     else:
#         logger.info("No meaningful correlations found.")

    


@task
def summary_report(df, correlations, tests):
    logger = get_run_logger()

    # Handle different country column names
    country_col = "Country name" if "Country name" in df.columns else df.columns[0]

    total_countries = df[country_col].nunique()
    total_years = df["year"].nunique()

    # Total countries and years
    logger.info(f"Total countries in the dataset: {total_countries}")
    logger.info(f"Total years included: {total_years}")

    # Regional averages
    region_means = df.groupby("Regional indicator")["Happiness score"].mean()

    top_regions = region_means.sort_values(ascending=False).head(3)
    bottom_regions = region_means.sort_values().head(3)

    logger.info("Top 3 happiest regions:")
    for region, score in top_regions.items():
        logger.info(f"{region}: {score:.3f}")

    logger.info("Bottom 3 happiest regions:")
    for region, score in bottom_regions.items():
        logger.info(f"{region}: {score:.3f}")

    # 2019 vs 2020 hypothesis test result
    logger.info("2019 vs 2020 Hypothesis Test:")
    logger.info(tests["covid_test"][2])

    # Strongest correlation after Bonferroni correction
    corrected_corrs = [c for c in correlations if c[4]]

    if corrected_corrs:
        strongest = max(corrected_corrs, key=lambda x: abs(x[1]))

        logger.info("Strongest correlation after Bonferroni correction:")
        logger.info(
            f"{strongest[0]} "
            f"(r = {strongest[1]:.3f}, p = {strongest[2]:.3e})"
        )
    else:
        logger.info("No correlations remained significant after Bonferroni correction.")

    logger.info("Summary report completed.")
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
