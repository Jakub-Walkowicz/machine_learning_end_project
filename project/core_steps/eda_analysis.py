from constants.constants import Columns
import numpy as np
import pandas as pd
from project.utils.charts import barplot_balance, heatmap_correlation, boxplot_histplot_outliers
from project.utils.utils import calc_class_distribution


from scipy.stats import trim_mean


def perform_eda_analysis(df):

    # Printing Top 3 Records
    print("Head -- \n", df.head(3))

    # Printing last 3 Records
    print("\n\nTail -- \n", df.tail(3))

    # Data description
    print("\n\nDescription -- \n", df.describe())

    # Data info
    print("\n\nInfo -- \n")
    df.info()

    # Check the number of null values
    print("\n\nNumber of null values -- \n\n", df.isnull().sum())

    # Analyse the 'jobs' column
    print(
        "\n\nJob type and its frequency in the dataset -- \n\n",
        df[Columns.JOB].value_counts(),
    )

    # Analyse the values of the 'age' column
    print(
        f"\n\nMinimum age value: {df[Columns.AGE].min()}, Maximum age value: {df[Columns.AGE].max()} -- \n\n"
    )

    # Analyse the impact of individual variables on the forecast variable Y
    df_copy = df.copy()
    df_copy[Columns.OUTPUT] = df_copy[Columns.OUTPUT].map({"yes": 1, "no": 0})
    correlations = df_copy.corr(numeric_only=True)
    correlations = correlations.sort_values(by=Columns.OUTPUT, ascending=False)
    # heatmap_correlation(correlations)
    print(
        "\n\nImpact of specific variables on the forecast variable y --\n\n",
        correlations,
    )

    # Assessment of the balance of the dataset
    y_value_counts = df[Columns.OUTPUT].value_counts().reset_index()
    print("\n\nThe balance of the dataset --\n\n", y_value_counts)
    # barplot_balance(y_value_counts)

    # Balance of the dataset in percentages
    total_samples = y_value_counts["count"].sum()
    class_distribution = calc_class_distribution(y_value_counts, total_samples)

    print(
        f"\nClass distribution: {class_distribution['majority']}% majority, {class_distribution['minority']}% minority"
    )

    # Detecting outliers in numeric variables
    # boxplot_histplot_outliers(df)
