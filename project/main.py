from project.constants.constants import Columns
from project.core_steps.eda_analysis import perform_eda_analysis
from project.core_steps.prepare_data import prepare_data
from project.utils.utils import get_df

# Read the dataset from a csv file
df = get_df
# Perform EDA
perform_eda_analysis(df)

# Prepare dataset
prepare_data(df)
