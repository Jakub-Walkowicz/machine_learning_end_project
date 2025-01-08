from constants.constants import Columns
import numpy as np
import pandas as pd
from project.utils.charts import (
    barplot_balance,
    heatmap_correlation,
    boxplot_histplot_outliers,
)
from project.utils.utils import calc_class_distribution, get_correlations


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

    # Column 'contact' analysis
    print("\n\nColumn 'contact' unique values: -- \n\n", df[Columns.CONTACT].unique())

    # Analyse the impact of individual variables on the forecast variable Y
    correlations = get_correlations(df)
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
    calc_class_distribution(y_value_counts, y_value_counts["count"].sum())

    # Detecting outliers in numeric variables
    # boxplot_histplot_outliers(df)
