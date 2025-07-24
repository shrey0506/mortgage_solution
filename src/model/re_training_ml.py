import os
import pickle
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier # Assuming LightGBM is one of your classification models
from sklearn.ensemble import RandomForestClassifier # Assuming RandomForest is another


def manage_classification_model(
    model_name: str,
    model_instance: object,
    X_train_processed: np.ndarray,
    y_train_cls: np.ndarray,
    X_test_processed: np.ndarray,
    y_test_cls: np.ndarray,
    accuracy_threshold: float = 0.80,
    model_filename: str = 'classification_model.pkl'
) -> object:
    """
    Checks the performance of a classification model. If accuracy is below a threshold,
    retrains the model. Otherwise, it attempts to load a saved model. If no saved
    model exists or loading fails, it trains a new model.

    Args:
        model_name (str): Name of the model (e.g., "LightGBM_Classifier").
        model_instance (object): An unfitted or fitted instance of the classification model.
        X_train_processed (np.ndarray): Preprocessed training features.
        y_train_cls (np.ndarray): Training labels.
        X_test_processed (np.ndarray): Preprocessed testing features.
        y_test_cls (np.ndarray): Testing labels.
        accuracy_threshold (float): The accuracy threshold for retraining.
        model_filename (str): The filename for saving/loading the model.

    Returns:
        object: A fitted classification model instance.
    """
    print(f"Managing {model_name}...")

    # Check if a saved model exists
    if os.path.exists(model_filename):
        print(f"Found saved model file: {model_filename}. Loading model...")
        try:
            with open(model_filename, 'rb') as f:
                loaded_model = pickle.load(f)
            print("Model loaded successfully.")

            # Evaluate the loaded model
            y_pred = loaded_model.predict(X_test_processed)
            accuracy = accuracy_score(y_test_cls, y_pred)
            print(f"Loaded model accuracy on test set: {accuracy:.4f}")

            # Decide whether to use the loaded model or retrain
            if accuracy >= accuracy_threshold:
                print(f"Loaded model meets accuracy threshold ({accuracy_threshold:.2f}). Using loaded model.")
                return loaded_model
            else:
                print(f"Loaded model accuracy ({accuracy:.4f}) is below threshold ({accuracy_threshold:.2f}). Retraining model...")
                # Proceed to train below

        except Exception as e:
            print(f"Error loading or evaluating saved model: {e}. Training a new model...")
            # Proceed to train below
    else:
        print(f"No saved model file found at {model_filename}. Training a new model...")
        # Proceed to train below

    # Train the model if no saved model was used or if retraining is needed
    print(f"Training {model_name}...")
    fitted_model = model_instance.fit(X_train_processed, y_train_cls)
    print(f"{model_name} trained.")

    # Evaluate the newly trained model
    y_pred_new = fitted_model.predict(X_test_processed)
    accuracy_new = accuracy_score(y_test_cls, y_pred_new)
    print(f"Newly trained model accuracy on test set: {accuracy_new:.4f}")

    # Save the newly trained model if its accuracy is above the threshold
    if accuracy_new >= accuracy_threshold:
        print(f"Newly trained model meets accuracy threshold ({accuracy_threshold:.2f}). Saving model to {model_filename}...")
        try:
            with open(model_filename, 'wb') as f:
                pickle.dump(fitted_model, f)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")
    else:
        print(f"Newly trained model accuracy ({accuracy_new:.4f}) is below threshold ({accuracy_threshold:.2f}). Not saving.")


    return fitted_model

# Example Usage (assuming models, X_train_processed, etc. are defined):
# trained_lgbm = manage_classification_model(
#     model_name="LightGBM_Classifier",
#     model_instance=LGBMClassifier(random_state=42), # Provide a new instance
#     X_train_processed=X_train_processed,
#     y_train_cls=y_train_cls,
#     X_test_processed=X_test_processed,
#     y_test_cls=y_test_cls,
#     model_filename='lgbm_classifier.pkl'
# )

# trained_rf = manage_classification_model(
#     model_name="RandomForest_Classifier",
#     model_instance=RandomForestClassifier(random_state=42), # Provide a new instance
#     X_train_processed=X_train_processed,
#     y_train_cls=y_train_cls,
#     X_test_processed=X_test_processed,
#     y_test_cls=y_test_cls,
#     model_filename='rf_classifier.pkl'
# )