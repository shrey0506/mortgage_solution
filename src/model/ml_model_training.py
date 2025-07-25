from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression  # Example classification model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Example models
from lightgbm import LGBMClassifier, LGBMRegressor  # Example models
from sklearn.model_selection import train_test_split  # Needed for splitting

class ClassificationPipeline:
    """
    Pipeline for the mortgage approval classification task.
    Includes preprocessing and a classification model.
    """
    def __init__(self, model, preprocessor):
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),  # Use the passed preprocessor
            ('classifier', model)
        ])

    def fit(self, X, y):
        """Fits the classification pipeline."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """Makes predictions using the classification pipeline."""
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Makes probability predictions using the classification pipeline."""
        return self.pipeline.predict_proba(X)

    def evaluate(self, X, y):
        """Evaluates the classification pipeline."""
        score = self.pipeline.score(X, y)
        print(f"Classification Pipeline Score: {score}")
        return score


class RegressionPipeline:
    """
    Pipeline for the interest rate regression task.
    Includes preprocessing and a regression model.
    """
    def __init__(self, model, preprocessor):
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),  # Use the passed preprocessor
            ('regressor', model)
        ])

    def fit(self, X, y):
        """Fits the regression pipeline."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """Makes predictions using the regression pipeline."""
        return self.pipeline.predict(X)

    def evaluate(self, X, y):
        """Evaluates the regression pipeline."""
        score = self.pipeline.score(X, y)  # R-squared for regression
        print(f"Regression Pipeline R-squared: {score}")
        return score

