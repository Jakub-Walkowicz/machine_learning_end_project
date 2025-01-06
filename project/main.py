import pandas as pd
import numpy as np
from project.constants.constants import Columns
from project.core_steps.eda_analysis import perform_eda_analysis

# Read the dataset from a csv file
df = pd.read_csv("data/bank_full.csv", sep=";")

# Perform EDA
perform_eda_analysis(df)


