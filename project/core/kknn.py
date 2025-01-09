from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from ceteris_paribus.explainer import explain
import shap
import pandas as pd


def build_kknn(X_train, y_train, X_test, y_test):
    # Defining parameters for the model
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["distance"],
        "p": [1, 2],
    }

    # Building the grid search
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring=["accuracy", "f1", "precision", "recall"],
        refit="f1",
        # verbose=2,
    )

    # Fitting the grid search to the training data
    grid_search.fit(X_train, y_train)
    print("\nBest parameters for KKNN: ", grid_search.best_params_)

    # Evaluating the model on the test data using the best parameters
    grid_predictions = grid_search.predict(X_test)
    print(
        "\n\nClassification report for KKNN model: -- \n\n",
        classification_report(y_test, grid_predictions),
    )

    # Creating and plotting a confusion matrix
    matrix = confusion_matrix(y_test, grid_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)

    matrix = confusion_matrix(y_test, grid_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)

    return disp
