from mortgage.src.processing.processing import run_pipeline
from mortgage.src.model import train_model, evaluate_model

def ml_model_main():
    """Main function to run the machine learning model pipeline."""
    # Step 1: Run the data processing pipeline
    processed_data = run_pipeline()

    # Step 2: Train the model
    model = train_model(processed_data)

    # Step 3: Evaluate the model
    evaluation_results = evaluate_model(model, processed_data)

    return evaluation_results


def run_pipeline_from_pkl():
    """Function to run the pipeline from a pre-trained model."""
    processed_data = run_pipeline()
    model = train_model(processed_data, load_from_pkl=True)
    evaluation_results = evaluate_model(model, processed_data)
    return evaluation_results