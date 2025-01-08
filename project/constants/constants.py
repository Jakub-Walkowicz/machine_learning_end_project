from project.constants.columns import Columns

FILE_PATH = "data/bank_full.csv"

COLUMNS_FOR_DUMMY_ENCODING = [
    Columns.JOB,
    Columns.MARITAL,
    Columns.EDUCATION,
    Columns.DEFAULT,
    Columns.HOUSING,
    Columns.LOAN,
    Columns.CONTACT,
    Columns.MONTH,
    Columns.POUTCOME,
]

NUMERIC_COLUMNS = [
    Columns.AGE,
    Columns.BALANCE,
    Columns.DAY,
    Columns.DURATION,
    Columns.CAMPAIGN,
    Columns.PDAYS,
    Columns.PREVIOUS,
]
