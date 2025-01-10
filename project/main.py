from project.constants.constants import Columns
from imblearn.pipeline import Pipeline
from project.core.eda_analysis import perform_eda_analysis
from project.utils.utils import (
    get_df,
    calc_class_distribution,
    compare_results,
)
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from project.core.kknn import build_kknn
from project.core.svm import build_svm
from project.core.bagging import build_bagging
from project.core.data_preparation import prepare_data

# Read the dataset from a csv file
df = get_df()
# Perform EDA
perform_eda_analysis(df)

# Encode variables
df = prepare_data(df)

# Split the dataset into X and y variables
X = df.drop(Columns.OUTPUT, axis=1)
y = df[Columns.OUTPUT]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create and fit pipeline
pipeline = Pipeline(
    [
        ("scaler", SklearnTransformerWrapper(StandardScaler())),
        ("undersampler", RandomUnderSampler(random_state=42)),
    ]
)

# Making sure that the training set distribution is the same as the entire dataset's distribution (approx. 88.3% major class vs 11.7% minor class)
print("\n\nTraining set distribution --")
calc_class_distribution(y_train.value_counts().reset_index(), y_train.count())

# Undersampling
X_train_p, y_train_p = pipeline.fit_resample(X_train, y_train)
print("\n\nTraining set distribution after undersampling --")
calc_class_distribution(y_train_p.value_counts().reset_index(), y_train_p.count())

# Transform test data (only scaling, without undersampling to preserve test set distribution)
X_test_p = pipeline["scaler"].transform(X_test)

# KKKN
disp_kknn = build_kknn(X_train_p, y_train_p, X_test_p, y_test)
# # SVM model
disp_svm = build_svm(X_train_p, y_train_p, X_test_p, y_test)

# # # Bagging
disp_bagging = build_bagging(X_train_p, y_train_p, X_test_p, y_test)

compare_results(disp_kknn, disp_svm, disp_bagging)
