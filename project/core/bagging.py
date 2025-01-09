from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier


def build_bagging(X_train, y_train, X_test, y_test):
    # Defining parameters for the model
    param_grid = {
        "n_estimators": [10, 50, 100],
        "max_samples": [0.5, 0.7, 1.0],
    }

    # Building the grid search
    grid_search = GridSearchCV(
        BaggingClassifier(DecisionTreeClassifier()),
        param_grid=param_grid,
        cv=5,
        scoring=["accuracy", "f1", "precision", "recall"],
        refit="f1",
        # verbose=2,
    )

    # Fitting the grid search to the training data
    grid_search.fit(X_train, y_train)
    # print("\nBest parameters for Bagging: ", grid_search.best_params_)

    # Evaluating the model on the test data using the best parameters
    grid_predictions = grid_search.predict(X_test)
    print(
        "\n\nClassification report for Bagging model: -- \n\n",
        classification_report(y_test, grid_predictions),
    )

    # Creating and plotting a confusion matrix
    matrix = confusion_matrix(y_test, grid_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    return disp
