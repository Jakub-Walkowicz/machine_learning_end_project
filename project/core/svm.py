from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay


def build_svm(X_train, y_train, X_test, y_test):
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": [0.1, 0.01, 0.001],
        "kernel": ["linear"],
    }

    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring=["accuracy", "f1", "precision", "recall"],
        refit="f1",
        # verbose=2,
    )

    grid_search.fit(X_train, y_test)
    # print("\nBest parameters for SVM: ", grid_search.best_params_)

    # Evaluating the model on the test data using the best parameters
    grid_predictions = grid_search.predict(X_test)
    print(
        "\n\nClassification report for SVM model: -- \n\n",
        classification_report(y_test, grid_predictions),
    )

    # Creating and plotting a confusion matrix
    matrix = confusion_matrix(y_test, grid_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    return disp
