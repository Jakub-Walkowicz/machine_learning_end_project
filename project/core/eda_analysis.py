from constants.constants import Columns, CATEGORICAL_COLUMNS
from project.utils.utils import (
    barplot_balance,
    heatmap_correlation,
    boxplot_histplot_outliers,
)
from project.utils.utils import calc_class_distribution, get_correlations
from scipy.stats import chi2_contingency
import pandas as pd


def analyze_categorical_relationships(df):

    results = {}
    for column in CATEGORICAL_COLUMNS:
        # Create contingency table
        contingency = pd.crosstab(df[column], df["y"])

        # Perform the test
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        results[column] = {"chi2": chi2, "p_value": p_value, "degrees_of_freedom": dof}

        print(f"\n\nChi-square test for variable {column}:")
        print(f"Chi2 statistic: {chi2:.2f}")
        print(f"P-value: {p_value:.10f}")

        # Interpretation
        if p_value < 0.05:
            print(
                f"Variable {column} has a statistically significant relationship with the target variable"
            )
        else:
            print(f"No statistically significant relationship for variable {column}")


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

    # Analyse the impact of variables on each other
    correlations = get_correlations(df)
    # heatmap_correlation(correlations)
    print(
        "\n\nImpact of specific variables on the forecast variable y --\n\n",
        correlations,
    )
    
    # Perform analysis of the categorical variables
    analyze_categorical_relationships(df)

    # Assessment of the balance of the dataset
    y_value_counts = df[Columns.OUTPUT].value_counts().reset_index()
    print("\n\nThe balance of the dataset --\n\n", y_value_counts)

    barplot_balance(y_value_counts)

    # # Calculate balance of the dataset in percentages
    calc_class_distribution(y_value_counts, y_value_counts["count"].sum())

    # # Detecting outliers in numeric variables
    boxplot_histplot_outliers(df)
