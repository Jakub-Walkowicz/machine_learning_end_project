from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from project.utils.utils import shap_interpretation
import shap


def build_svm(X_train, y_train, X_test, y_test):
    param_grid = {
        # "C": [0.1, 1, 10],
        # "gamma": [0.1, 0.01, 0.001],
        "C": [0.1],
        "gamma": [0.1],
        "kernel": ["linear"],
    }

    grid = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring=["accuracy", "f1", "precision", "recall"],
        refit="f1",
        # verbose=2,
    )

    grid.fit(X_train, y_train)
    # print("\nBest parameters for SVM: ", grid.best_params_)

    # Evaluating the model on the test data using the best parameters
    grid_pred = grid.predict(X_test)
    print(
        "\n\nClassification report for SVM model: -- \n\n",
        classification_report(y_test, grid_pred),
    )

    # Creating and plotting a confusion matrix
    matrix = confusion_matrix(y_test, grid_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)

    # ------------------------------------------- #
    # SHAP #

    model = grid.best_estimator_
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    shap.dependence_plot("balance", shap_values, X_test)

    # shap_interpretation()
    return disp
