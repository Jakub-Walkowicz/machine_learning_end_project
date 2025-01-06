# Calculate class distribution based on provided 'value_counts()' dataframe ('y_value_counts') and total number of records ('total_samples')
from constants.columns import Columns
def calc_class_distribution(y_value_counts, total_samples):
    return {
        "majority": round((y_value_counts.iloc[0]["count"] / total_samples) * 100, 2),
        "minority": round((y_value_counts.iloc[1]["count"] / total_samples) * 100, 2),
    }

def get_correlations(df):
    df_copy = df.copy()
    df_copy[Columns.OUTPUT] = df_copy[Columns.OUTPUT].map({"yes": 1, "no": 0})
    correlations = df_copy.corr(numeric_only=True)
    return correlations.sort_values(by=Columns.OUTPUT, ascending=False)