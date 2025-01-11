from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from project.constants.constants import SAMPLE_SIZE
from sklearn.metrics import ConfusionMatrixDisplay
import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def build_svm(X_train, y_train, X_test, y_test):
    param_grid = {
        "C": [1, 10],
        "kernel": ["linear"],
    }

    # Initialize SVC
    grid = GridSearchCV(
        SVC(probability=True),
        param_grid,
        cv=3,
        scoring="f1",
        refit="f1",
        verbose=2,
    )

    grid.fit(X_train, y_train)
    grid_pred = grid.predict(X_test)
    print("\nClassification report:\n", classification_report(y_test, grid_pred))

    # Confusion matrix
    matrix = confusion_matrix(y_test, grid_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)

    # SHAP analysis
    # Create background data for SHAP
    background = shap.sample(X_train, SAMPLE_SIZE)
    
    # Create explainer
    explainer = shap.LinearExplainer(
        grid.best_estimator_, 
        background,
        feature_names=X_test.columns.tolist()
    )

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test.iloc[:SAMPLE_SIZE])

    # Create and show summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test.iloc[:SAMPLE_SIZE],
        feature_names=X_test.columns,
        show=False
    )
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

    # Create and show force plot for first prediction
    plt.figure(figsize=(15, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_test.iloc[0],
        feature_names=X_test.columns,
        matplotlib=True,
        show=False
    )
    plt.title("SHAP Force Plot for First Prediction")
    plt.tight_layout()
    plt.show()

    # Calculate and print feature importance
    feature_importance = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': feature_importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(importance_df.head(10))

    return disp