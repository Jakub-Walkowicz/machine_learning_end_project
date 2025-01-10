import pandas as pd
from project.constants.constants import COLUMNS_FOR_DUMMY_ENCODING
from constants.columns import Columns


# Preprocesses the data by encoding categorical variables and mapping the target variable to binary values (0 and 1)
def prepare_data(df):
    df = pd.get_dummies(
        df, columns=COLUMNS_FOR_DUMMY_ENCODING, drop_first=True, dtype=int
    )
    target_map = {"yes": 1, "no": 0}
    df[Columns.OUTPUT] = df[Columns.OUTPUT].map(target_map)

    # Feature engineering
    # Creation of a new feature based on 'previous' variable
    df["previous_binary"] = df[Columns.PREVIOUS].apply(lambda x: 1 if x > 0 else 0)

    return df
