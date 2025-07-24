from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression # Example classification model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # Example models
from lightgbm import LGBMClassifier, LGBMRegressor # Example models
from sklearn.model_selection import train_test_split # Needed for splitting

# Assuming 'full_preprocessor_pipeline' is defined in a previous cell

class ClassificationPipeline:
    """
    Pipeline for the mortgage approval classification task.
    Includes preprocessing and a classification model.
    """
    def __init__(self, model):
        self.pipeline = Pipeline(steps=[
            ('preprocessor', full_preprocessor_pipeline), # Use the defined preprocessor
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
        # This is a placeholder; actual evaluation metrics would be calculated here
        score = self.pipeline.score(X, y)
        print(f"Classification Pipeline Score: {score}")
        return score


class RegressionPipeline:
    """
    Pipeline for the interest rate regression task.
    Includes preprocessing and a regression model.
    """
    def __init__(self, model):
        self.pipeline = Pipeline(steps=[
            ('preprocessor', full_preprocessor_pipeline), # Use the defined preprocessor
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
        # This is a placeholder; actual evaluation metrics would be calculated here
        score = self.pipeline.score(X, y) # R-squared for regression
        print(f"Regression Pipeline R-squared: {score}")
        return score


# Example Usage (assuming X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg are defined):

# Instantiate models
# classification_model = LGBMClassifier(random_state=42)
# regression_model = RandomForestRegressor(random_state=42)

# Create pipelines
# classification_pipeline = ClassificationPipeline(model=classification_model)
# regression_pipeline = RegressionPipeline(model=regression_model)

# Fit pipelines (example)
# classification_pipeline.fit(X_train, y_train_cls)
# regression_pipeline.fit(X_train, y_train_reg)

# Evaluate pipelines (example)
# classification_pipeline.evaluate(X_test, y_test_cls)
# regression_pipeline.evaluate(X_test, y_test_reg)