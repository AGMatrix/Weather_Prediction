"""
Lab 2 - Weather Prediction
Main training script for model creation and evaluation.
"""
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_collection import collect_historical_cf6_data, collect_daily_cf6_data
from data_processing import build_daily_dataset, prepare_training_data
from models import train_models, evaluate_models


def main():
    """Main function to run the training process."""
    print("Weather Prediction Training Script")
    print("==================================")

    # Define city mappings for data collection
    city_mappings = {
        # New York State
        "ROC": "BUF",  # Rochester, NY → Buffalo NWS Office
        "BUF": "BUF",  # Buffalo, NY → Buffalo NWS Office
        "SYR": "BGM",  # Syracuse, NY → Binghamton NWS Office
        "ALB": "ALY",  # Albany, NY → Albany NWS Office

        # New England
        "BOS": "BOX",  # Boston, MA → Boston/Norton NWS Office
        "BTV": "BTV",  # Burlington, VT → Burlington NWS Office
        "PWM": "GYX",  # Portland, ME → Gray NWS Office
        "BDL": "BOX",  # Hartford, CT → Boston/Norton NWS Office

        # Midwest
        "DTW": "DTX",  # Detroit, MI → Detroit NWS Office
        "CLE": "CLE",  # Cleveland, OH → Cleveland NWS Office
        "ORD": "LOT",  # Chicago, IL (O'Hare) → Chicago NWS Office
        "MSP": "MPX",  # Minneapolis, MN → Twin Cities NWS Office
        "MKE": "MKX",  # Milwaukee, WI → Milwaukee NWS Office
        "IND": "IND",  # Indianapolis, IN → Indianapolis NWS Office

        # Pennsylvania/Ohio Valley
        "PIT": "PBZ",  # Pittsburgh, PA → Pittsburgh NWS Office
        "ERI": "CLE",  # Erie, PA → Cleveland NWS Office
        "PHL": "PHI",  # Philadelphia, PA → Philadelphia NWS Office
        "CVG": "ILN",  # Cincinnati, OH → Wilmington, OH NWS Office

        # Mid-Atlantic
        "IAD": "LWX",  # Washington, DC (Dulles) → Baltimore/Washington NWS Office
    }

    # Data directories
    data_dir = "data"
    models_dir = "models"
    daily_data_dir = "daily_data"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(daily_data_dir, exist_ok=True)

    # Check if we need to collect data
    files = os.listdir(data_dir) if os.path.exists(data_dir) else []
    cf6_files = [f for f in files if f.endswith('.txt')]

    if len(cf6_files) == 0:
        print("No data found. Collecting historical weather data...")

        start_date = datetime(2020, 1, 1).date()
        end_date = datetime(2025, 3, 31).date()
        collect_historical_cf6_data(city_mappings, start_date, end_date)

        # Collect recent daily data for prediction
        today = datetime.now().date()
        first_of_month = today.replace(day=1)  # Get the 1st day of current month

        # Iterate through all days from 1st of month to yesterday
        current_date = first_of_month
        while current_date <= today:
            collect_daily_cf6_data(city_mappings, current_date, daily_data_dir)
            current_date += timedelta(days=1)

        # Check if we got any data after collection
        files = os.listdir(data_dir)
        cf6_files = [f for f in files if f.endswith('.txt')]

        if len(cf6_files) == 0:
            print("ERROR: No data was collected. Cannot proceed with model training.")
            return  # Exit the function
    else:
        print(f"Found {len(cf6_files)} data files in {data_dir}. Skipping data collection.")

    # Process the collected data
    print("Processing data and building features...")
    df = build_daily_dataset(data_dir, list(city_mappings.keys()))

    # Check if we have data and required columns
    if df.empty:
        print("ERROR: No data was processed. Cannot proceed with model training.")
        return

    required_columns = ['avg_temp', 'max_temp', 'min_temp', 'precipitation',
                       'avg_wind_speed', 'wind_direction', 'max_wind_gust', 'date']
    for col in required_columns:
        if col not in df.columns:
            print(f"ERROR: Missing required column {col} in dataset")
            return

    # Display some statistics about the data
    print(f"Data shape: {df.shape}")
    print(f"Cities: {df['city'].unique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Prepare features and targets for modeling
    print("Preparing training data...")
    X, y = prepare_training_data(df)
    print(f"Feature matrix shape: {X.shape}")

    # Check if we have enough data to train
    if X.shape[0] < 10:  # Arbitrary minimum number of samples
        print("ERROR: Not enough data to train models.")
        return


    n_samples = X.shape[0]
    n_train = int(0.8 * n_samples)  # 80% for training
    X_train, X_test = X[:n_train], X[n_train:]
    y_train = {target: values[:n_train] for target, values in y.items()}
    y_test = {target: values[n_train:] for target, values in y.items()}

    # Train the models
    print("Training models...")
    models = train_models(X_train, y_train, output_dir=models_dir, param_search=True)

    # Evaluate the models
    print("Evaluating models...")
    results = evaluate_models(models, X_test, y_test)

    # Plot feature importances
    print("Plotting feature importance...")
    feature_names = [
        "Avg temp delta",
        "Max temp delta",
        "Min temp delta",
        "Yesterday precipitation",
        "3-day precipitation",
        "Wind speed delta",
        "Wind direction sine",
        "Wind direction cosine",
        "Max wind gust",
        "Spring", "Summer", "Fall", "Winter",
        # Western cities wind components
        "DTW wind dir", "DTW wind speed",
        "CLE wind dir", "CLE wind speed",
        "ORD wind dir",
        "MSP wind dir",
        "ERI wind dir",
        "BUF wind dir", "BUF wind speed",
        "PIT wind dir"
    ]

    # Ensure we have the right number of feature names
    if len(feature_names) < X.shape[1]:
        feature_names.extend([f"Feature {i}" for i in range(len(feature_names), X.shape[1])])

    # Create individual plots for each model
    target_names = ['temp_higher', 'above_avg', 'precipitation']
    model_types = ['besttree', 'bestforest']

    for i, target_name in enumerate(target_names):
        for j, model_type in enumerate(model_types):
            model_key = f'{model_type}_{target_name}'
            if model_key in models:
                model = models[model_key]

                # Create a new figure for each model
                plt.figure(figsize=(10, 8))

                # Sort feature importances (show top 15)
                indices = np.argsort(model.feature_importances_)[-15:]

                # Plot
                plt.barh(range(len(indices)), model.feature_importances_[indices], align='center')
                plt.yticks(range(len(indices)), [feature_names[idx] for idx in indices])
                plt.title(f'{model_type.capitalize()} - {target_name.replace("_", " ").title()}')
                plt.xlabel('Feature Importance')
                plt.tight_layout()

                # Save individual plot
                plt.savefig(os.path.join(models_dir, f'feature_importance_{model_type}_{target_name}.png'))
                plt.close()

    print("Training complete! Models saved to", models_dir)


if __name__ == "__main__":
    main()