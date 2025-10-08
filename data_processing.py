"""
Lab 2 - Weather Prediction
Module for extracting data from CF6 reports and building features.
"""
import re
import numpy as np
import pandas as pd
import os
import math
from datetime import datetime, timedelta


def build_daily_dataset(data_dir, city_codes):
    """
    Build a dataset from CF6 report files for multiple cities.

    Args:
        data_dir (str): Directory containing CF6 report files
        city_codes (list): List of city codes to include in the dataset

    Returns:
        pandas.DataFrame: DataFrame containing the combined data
    """
    all_data = []
    print(f"Processing files from {data_dir} for cities: {city_codes}")

    # List all files in the data directory
    files = os.listdir(data_dir)
    print(f"Found {len(files)} files in directory")

    for file in files:
        # Check if the file is for one of our target cities
        city_match = None
        for city in city_codes:
            if file.startswith(f"{city}_"):
                city_match = city
                break

        if city_match:
            # Extract year and month from filename
            parts = file.split('_')
            if len(parts) < 3:
                print(f"Skipping file with unexpected format: {file}")
                continue

            try:
                year = int(parts[1])
                month = int(parts[2].split('.')[0])
            except (ValueError, IndexError) as e:
                print(f"Error parsing year/month from {file}: {e}")
                continue

            # Read the CF6 report
            try:
                file_path = os.path.join(data_dir, file)
                print(f"Processing file: {file_path}")
                with open(file_path, 'r') as f:
                    cf6_text = f.read()

                # Extract data from the report using a more robust approach
                daily_data = []

                # Split the whole file into lines — each line might be for one day of weather.
                lines = cf6_text.split('\n')

                for line in lines:
                    # Look for lines that contain day numbers (1-31)
                    day_match = re.match(r'^\s*(\d{1,2})\s', line)

                    if day_match:
                        try:
                            # Confirm this is a data line with a valid day number
                            day_num = int(day_match.group(1))
                            if 1 <= day_num <= 31:
                                # Split by whitespace
                                values = line.strip().split()

                                # Skip if this is a header line with letters or M- missing Value, T-Trace Amount
                                if any(v for v in values if re.search(r'[A-Za-z]', v) and v not in ['M', 'T']):
                                    continue

                                # Initialize day_data
                                day_data = {
                                    'day': day_num,
                                }

                                # Ensure we have enough values for a valid data line with base data
                                if len(values) >= 10:
                                    try:
                                        # Add basic weather data
                                        day_data.update({
                                            'max_temp': float(values[1]) if values[1] != 'M' else None,
                                            'min_temp': float(values[2]) if values[2] != 'M' else None,
                                            'avg_temp': float(values[3]) if values[3] != 'M' else None,
                                            'departure': float(values[4]) if values[4] != 'M' else None,
                                            'hdd': float(values[5]) if values[5] != 'M' else None,
                                            'cdd': float(values[6]) if values[6] != 'M' else None,
                                            'precipitation': float(values[7]) if values[7] != 'M' and values[7] != 'T'
                                                else 0.001 if values[7] == 'T' else None,
                                            'snow': float(values[8]) if values[8] != 'M' and values[8] != 'T'
                                                else 0.001 if values[8] == 'T' else None,
                                            'snow_depth': float(values[9]) if values[9] != 'M' and values[9] != 'T'
                                                else 0.001 if values[9] == 'T' else None,
                                        })

                                        # Add wind data if available
                                        if len(values) >= 13:
                                            day_data.update({
                                                'avg_wind_speed': float(values[10]) if values[10] != 'M' else None,
                                                'max_wind_gust': float(values[11]) if values[11] != 'M' else None,
                                                'wind_direction': int(values[12]) if values[12] != 'M' else None,
                                            })

                                        # Add city and date information
                                        day_data['city'] = city_match
                                        day_data['year'] = year
                                        day_data['month'] = month
                                        day_data['date'] = datetime(year, month, day_num)

                                        daily_data.append(day_data)
                                    except Exception as e:
                                        print(f"Error parsing values for day {day_num}: {e}")
                                        continue
                        except Exception as e:
                            print(f"Error processing line: {line.strip()}")
                            print(f"Exception: {e}")
                            continue

                all_data.extend(daily_data)
                print(f"Extracted {len(daily_data)} days of data from file {file}")

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

    # Convert to DataFrame
    print(f"Total data points collected: {len(all_data)}")
    df = pd.DataFrame(all_data)

    # Make sure we have data before sorting
    if df.empty:
        print("Warning: No data was processed. DataFrame is empty.")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['city', 'date', 'day', 'max_temp', 'min_temp', 'avg_temp',
                                     'departure', 'hdd', 'cdd', 'precipitation',
                                     'snow', 'snow_depth', 'avg_wind_speed', 'max_wind_gust',
                                     'wind_direction', 'year', 'month'])

    # Sort by date
    df = df.sort_values(['city', 'date'])
    print(f"Created DataFrame with shape: {df.shape}")

    return df


def prepare_training_data(df, lookback_days=5):
    """
    Prepare training data for the models.

    Args:
        df (pandas.DataFrame): DataFrame containing weather data
        lookback_days (int): Number of days to look back for features

    Returns:
        tuple: X (features) and y (targets) for training
    """
    X = []
    y_temp_higher = []
    y_above_avg = []
    y_precipitation = []

    # Get unique dates in the dataset
    unique_dates = sorted(df['date'].unique())

    for i in range(lookback_days, len(unique_dates)):
        target_date = unique_dates[i]

        # Get data for the target date and preceding days
        date_range = unique_dates[i - lookback_days:i + 1]
        period_data = df[df['date'].isin(date_range)]

        # Build features from the lookback period
        features = []

        # Get Rochester data for the lookback period
        roc_data = period_data[period_data['city'] == 'ROC'].sort_values('date')

        if len(roc_data) >= lookback_days:
            # Temperature deltas (day-to-day changes)
            if 'avg_temp' in roc_data.columns:
                # Use the most recent days for delta calculation
                recent_temps = roc_data['avg_temp'].tail(2).values#Gets the last 2 rows of this column (most recent 2 days in the lookback window)
                if len(recent_temps) == 2:
                    avg_temp_delta = recent_temps[1] - recent_temps[0]
                    features.append(avg_temp_delta)  # Avg temp delta
                else:
                    features.append(0)  # Default if not enough data
            else:
                features.append(0)

            # Max and min temp deltas
            if 'max_temp' in roc_data.columns and 'min_temp' in roc_data.columns:
                recent_max = roc_data['max_temp'].tail(2).values
                recent_min = roc_data['min_temp'].tail(2).values

                if len(recent_max) == 2 and len(recent_min) == 2:
                    max_temp_delta = recent_max[1] - recent_max[0]
                    min_temp_delta = recent_min[1] - recent_min[0]
                    features.append(max_temp_delta)  # Max temp delta
                    features.append(min_temp_delta)  # Min temp delta
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])

            # Precipitation features
            if 'precipitation' in roc_data.columns:
                # Recent precipitation
                recent_precip = roc_data['precipitation'].tail(1).values
                if len(recent_precip) == 1:
                    features.append(recent_precip[0])  # Yesterday's precipitation
                else:
                    features.append(0)

                # 3-day total precipitation
                precip_3day = roc_data['precipitation'].tail(3).sum()
                features.append(precip_3day)  # 3-day precipitation
            else:
                features.extend([0, 0])

            # Wind features
            # Wind speed delta
            if 'avg_wind_speed' in roc_data.columns:
                recent_wind = roc_data['avg_wind_speed'].tail(2).values
                if len(recent_wind) == 2 and not np.isnan(recent_wind).any():
                    wind_speed_delta = recent_wind[1] - recent_wind[0]
                    features.append(wind_speed_delta)  # Wind speed delta
                else:
                    features.append(0)
            else:
                features.append(0)

            # Wind direction as sine and cosine components (circular feature)
            if 'wind_direction' in roc_data.columns:
                recent_dir = roc_data['wind_direction'].tail(1).values
                if len(recent_dir) == 1 and recent_dir[0] is not None and not np.isnan(recent_dir[0]):
                    # Convert to radians
                    dir_rad = math.radians(recent_dir[0])
                    # Add sine and cosine components (preserves the circular nature)
                    features.append(math.sin(dir_rad))  # Wind direction sine
                    features.append(math.cos(dir_rad))  # Wind direction cosine
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])

            # Maximum wind gust
            if 'max_wind_gust' in roc_data.columns:
                recent_gust = roc_data['max_wind_gust'].tail(1).values
                if len(recent_gust) == 1 and recent_gust[0] is not None and not np.isnan(recent_gust[0]):
                    features.append(recent_gust[0])  # Recent max gust
                else:
                    features.append(0)
            else:
                features.append(0)

            # Add seasonal indicators
            # Get month from the most recent date
            month = roc_data['date'].iloc[-1].month
            features.append(1 if 3 <= month <= 5 else 0)  # Spring
            features.append(1 if 6 <= month <= 8 else 0)  # Summer
            features.append(1 if 9 <= month <= 11 else 0)  # Fall
            features.append(1 if month == 12 or month <= 2 else 0)  # Winter

            # Process data from other cities to build features
            western_cities = ["DTW", "CLE", "ORD", "MSP", "ERI", "BUF", "PIT"]
            for city in western_cities:
                city_data = period_data[period_data['city'] == city].sort_values('date')

                #Temperature and precipitation from other cities add more noise than any signal
                # # Temperature from this city
                # if not city_data.empty and 'avg_temp' in city_data.columns:
                #     features.append(city_data['avg_temp'].iloc[-1])  # Latest avg temp
                # else:
                #     features.append(0)
                #
                # # Precipitation from this city
                # if not city_data.empty and 'precipitation' in city_data.columns:
                #     features.append(city_data['precipitation'].iloc[-1])  # Latest precipitation
                # else:
                #     features.append(0)

                # Wind direction from this city (as easterly/westerly component)
                if not city_data.empty and 'wind_direction' in city_data.columns:
                    if not pd.isna(city_data['wind_direction'].iloc[-1]):
                        # Convert to radians and get cosine (easterly/westerly component)
                        dir_rad = math.radians(city_data['wind_direction'].iloc[-1])
                        features.append(math.cos(dir_rad))  # Easterly/westerly component
                    else:
                        features.append(0)
                else:
                    features.append(0)

                if city in ["BUF", "DTW", "CLE"] and 'avg_wind_speed' in city_data.columns:
                    features.append(city_data['avg_wind_speed'].iloc[-1])
                else:
                    features.append(0)

            # If we have all the features, add to training data
            if len(features) > 15:  # Increased minimum required features
                X.append(features)

                # Get target data (Rochester on target date)
                roc_target = df[(df['date'] == target_date) & (df['city'] == 'ROC')]

                if not roc_target.empty:
                    # Target 1: Is temperature higher than yesterday?
                    yesterday = df[(df['date'] == unique_dates[i - 1]) & (df['city'] == 'ROC')]
                    if not yesterday.empty and 'avg_temp' in roc_target.columns and 'avg_temp' in yesterday.columns:
                        y_temp_higher.append(roc_target['avg_temp'].iloc[0] > yesterday['avg_temp'].iloc[0])
                    else:
                        y_temp_higher.append(False)

                    # Target 2: Is temperature above average?
                    if 'departure' in roc_target.columns:
                        y_above_avg.append(roc_target['departure'].iloc[0] > 0)
                    else:
                        y_above_avg.append(False)

                    # Target 3: Is there precipitation?
                    if 'precipitation' in roc_target.columns:
                        y_precipitation.append(roc_target['precipitation'].iloc[0] > 0.01)
                    else:
                        y_precipitation.append(False)
                else:
                    # Default values if target data is missing
                    y_temp_higher.append(False)
                    y_above_avg.append(False)
                    y_precipitation.append(False)

    return np.array(X), {
        'temp_higher': np.array(y_temp_higher),
        'above_avg': np.array(y_above_avg),
        'precipitation': np.array(y_precipitation)
    }