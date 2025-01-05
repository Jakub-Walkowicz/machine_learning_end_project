from constants.constants import Columns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from scipy.stats import trim_mean


def perform_eda_analysis(df):

    # Printing Top 3 Records
    print("Head -- \n", df.head(3))

    # Printing last 3 Records
    print("\n\n Tail -- \n", df.tail(3))

    # Data description
    print("\n\n Description -- \n", df.describe())

    # Data info
    print("\n\n Info -- \n")
    df.info()

    # Check the number of null values
    print("\n\n Number of null values -- \n\n", df.isnull().sum())
    
    # Analyse the 'jobs' column
    print(
        "\n\n Job type and its frequency in the dataset -- \n\n",
        df[Columns.JOB].value_counts(),
    )



    # WYDRUKOWAC MIN I MAX DLA 'age'
    
    # wykres lokaty per wiek -- UPIEKSZYC WYKRES
    ax = sns.countplot(x="age", hue="y", data=df)
    ax.xaxis.set_major_locator(ticker.LinearLocator(10))
    plt.show()