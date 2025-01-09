from constants.columns import Columns
from constants.constants import FILE_PATH
import pandas as pd
from project.constants.constants import COLUMNS_FOR_DUMMY_ENCODING
import seaborn as sns
import matplotlib.pyplot as plt
from constants.constants import NUMERIC_COLUMNS


# Load the dataset from a csv file and return a pandas DataFrame
def get_df():
    return pd.read_csv(FILE_PATH, sep=";")


# Calculate class distribution based on provided 'value_counts()' dataframe ('y_value_counts') and total number of records ('total_samples')
def calc_class_distribution(y_value_counts, total_samples):
    class_distribution = {
        "majority": round((y_value_counts.iloc[0]["count"] / total_samples) * 100, 2),
        "minority": round((y_value_counts.iloc[1]["count"] / total_samples) * 100, 2),
    }
    print(
        f"\nClass distribution: {class_distribution['majority']}% majority, {class_distribution['minority']}% minority"
    )


# Calculates and prints the correlation between the numeric variables and the target variable (y)
def get_correlations(df):
    df_copy = df.copy()
    df_copy[Columns.OUTPUT] = df_copy[Columns.OUTPUT].map({"yes": 1, "no": 0})
    correlations = df_copy.corr(numeric_only=True)
    return correlations.sort_values(by=Columns.OUTPUT, ascending=False)


# Preprocesses the data by encoding categorical variables and mapping the target variable to binary values (0 and 1)
def prepare_data(df):
    df = pd.get_dummies(
        df, columns=COLUMNS_FOR_DUMMY_ENCODING, drop_first=True, dtype=int
    )
    target_map = {"yes": 1, "no": 0}
    df[Columns.OUTPUT] = df[Columns.OUTPUT].map(target_map)
    return df


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


# Draws boxplots and histograms for the numeric variables to detect outliers and visualize the distribution
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


# Compares the confusion matrices of the three models (KNN, SVM, Bagging) and visualizes them using a stacked bar plot
def compare_results(disp_kknn, disp_svm, disp_bagging):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    disp_kknn.plot(ax=ax1)
    ax1.set_title("KKNN Model")
    disp_svm.plot(ax=ax2)
    ax2.set_title("SVM Model")
    disp_bagging.plot(ax=ax3)
    ax3.set_title("Bagging Model")
    plt.show()
