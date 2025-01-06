# Calculate class distribution based on provided 'value_counts()' dataframe ('y_value_counts') and total number of records ('total_samples')
def calc_class_distribution(y_value_counts, total_samples):
    return {
        "majority": round((y_value_counts.iloc[0]["count"] / total_samples) * 100, 2),
        "minority": round((y_value_counts.iloc[1]["count"] / total_samples) * 100, 2),
    }
