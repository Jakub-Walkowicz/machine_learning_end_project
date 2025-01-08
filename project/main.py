from project.constants.constants import Columns
from project.core.eda_analysis import perform_eda_analysis
from project.utils.utils import get_df, prepare_data, calc_class_distribution
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from project.core.kknn import build_kknn

# Read the dataset from a csv file
df = get_df()
# Perform EDA
# perform_eda_analysis(df)

# Encode variables
df = prepare_data(df)

# Split the dataset into X and y variables
X = df.drop(Columns.OUTPUT, axis=1)
y = df[Columns.OUTPUT]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Making sure that the training set distribution is the same as the entire dataset's distribution (approx. 88.3% major class vs 11.7% minor class)
# calc_class_distribution(y_train.value_counts().reset_index(), y_train.count())

# Undersampling
undersample = RandomUnderSampler(random_state=42)
X_train_undersampled, y_train_undersampled = undersample.fit_resample(X_train, y_train)

# Standarisation
scaler = StandardScaler().fit(X_train_undersampled)
X_train_scaled = scaler.transform(X_train_undersampled)
X_test_scaled = scaler.transform(X_test)

# KKKN
kknn = build_kknn(X_train_scaled, y_train_undersampled, X_test_scaled, y_test)

# Decision tree


# SVM model
