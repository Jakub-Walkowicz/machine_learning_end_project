import pandas as pd
import numpy as np
from project.constants.constants import CATEGORICAL_COLUMNS, Columns
from eda_analysis import perform_eda_analysis

# Read the dataset from a csv file
df = pd.read_csv("data/bank_sample.csv", sep=";")

# Replace the value of "unknown" in categorical columns with np.nan for easier data analysis
df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].replace("unknown", np.nan)

# According to the point no. 7 from bank_description.txt the column 'durations' is disregarded
df = df.drop(labels=Columns.DURATION, axis=1)

# Perform EDA
perform_eda_analysis(df)
# print(df.columns)



# Clean the possible mistake with 'admin.' value in column 'job'
# 
