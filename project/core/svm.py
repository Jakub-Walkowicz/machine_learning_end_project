from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def build_svm(X_train_scaled, y_train_undersampled, X_test_scaled, y_test):
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": [0.1, 0.01, 0.001],
        "kernel": ["rbf"],
    }

    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring=["accuracy", "f1", "precision", "recall"],
        refit="f1",
        verbose=2
    )

    grid_search.fit(X_train_scaled, y_train_undersampled)
    print("\nBest parameters for SVM: ", grid_search.best_params_)

    grid_predictions = grid_search.predict(X_test_scaled)
    print(classification_report(y_test, grid_predictions))

    matrix = confusion_matrix(y_test, grid_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.show()
