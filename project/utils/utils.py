from constants.columns import Columns
from constants.constants import FILE_PATH
import pandas as pd
from project.constants.constants import COLUMNS_FOR_DUMMY_ENCODING


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


def get_correlations(df):
    df_copy = df.copy()
    df_copy[Columns.OUTPUT] = df_copy[Columns.OUTPUT].map({"yes": 1, "no": 0})
    correlations = df_copy.corr(numeric_only=True)
    return correlations.sort_values(by=Columns.OUTPUT, ascending=False)


def prepare_data(df):
    df = pd.get_dummies(
        df, columns=COLUMNS_FOR_DUMMY_ENCODING, drop_first=True, dtype=int
    )
    target_map = {"yes": 1, "no": 0}
    df[Columns.OUTPUT] = df[Columns.OUTPUT].map(target_map)
    return df
