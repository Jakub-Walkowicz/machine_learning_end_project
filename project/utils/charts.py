import seaborn as sns
import matplotlib.pyplot as plt
from constants.columns import Columns
from constants.constants import NUMERIC_COLUMNS


# Draws a barplot showing the balance of the dataset (variable 'y')
def barplot_balance(y_value_counts):
    sns.barplot(data=y_value_counts, x=Columns.OUTPUT, y="count")
    plt.title("Balance of the dataset")
    plt.show()


# Draws a heatmap showing the correlation between the numeric variables and the output variable
def heatmap_correlation(correlations):
    sns.heatmap(correlations)
    plt.title("Impact of specific variables on the forecast variable y")
    plt.show()


def boxplot_histplot_outliers(df):
    for column in NUMERIC_COLUMNS:
        plt.figure(figsize=(13, 8))
        plt.subplot(1, 2, 1)
        plt.title(f"Boxplot for {column}")
        sns.boxplot(data=df, x=df[column])
        
        plt.subplot(1, 2, 2)
        plt.title(f"Histogram for {column}")
        sns.histplot(data=df, x=df[column])
        plt.show()
