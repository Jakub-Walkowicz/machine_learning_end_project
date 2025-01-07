import pandas as pd
from project.constants.columns import Columns


# According to analysis from eda_analysis.py - there are no null values in the entire dataset
# According to analysis from eda_analysis.py - there are no outliers in the entire dataset
# Encoding variables - below


def prepare_data(df):
    # Encode variable 'age'
    df = pd.get_dummies(df, df[Columns.AGE], drop_first=True)
