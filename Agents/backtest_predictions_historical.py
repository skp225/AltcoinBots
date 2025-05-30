#!/usr/bin/env python3
"""
Historical Backtest Predictions

This script automates the process of:
1. Finding all historical prediction data files
2. Getting historical price data for the cryptocurrencies
3. Comparing the predictions with the actual prices at the appropriate times
4. Scoring the predictions based on accuracy
5. Storing the results in JSON and CSV formats

The script will:
- Look for all prediction data files in the prediction_data folder
- Get historical price data for the cryptocurrencies in those predictions
- Score each prediction (2 points if actual price crossed predicted price,
  1 point if within 10%, 0 points otherwise)
- Store the backtesting results as both JSON and CSV files
- Provide comprehensive analysis of prediction accuracy over time
"""

import os
import sys
import json
import glob
import csv
import subprocess
import importlib.util
from datetime import datetime, timedelta
import time
import traceback

# Try to import required packages, install if missing
try:
    import pandas as pd
    import requests
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Attempting to install dependencies...")
    
    # Try to run the dependency installer
    try:
        # Get the path to the install_dependencies.py script
        installer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "install_dependencies.py")
        
        if os.path.exists(installer_path):
            subprocess.run([sys.executable, installer_path], check=True)
            print("Dependencies installed successfully. Importing again...")
            
            # Try importing again
            import pandas as pd
            import requests
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
        else:
            print(f"Dependency installer not found at {installer_path}")
            print("Please install required packages manually: pandas, requests, numpy, matplotlib")
            subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "requests", "numpy", "matplotlib"], check=True)
            import pandas as pd
            import requests
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
    except Exception as install_error:
        print(f"Failed to install dependencies: {install_error}")
        print("Please install required packages manually: pandas, requests, numpy, matplotlib")
        sys.exit(1)

# Add parent directory to path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Telegram configuration
TELEGRAM_TOKEN = '7208093088:AAEyAaIf__zr3QqW0UbWrhE6MSO6MRtH2Dg'
CHAT_ID = '310580895'

def send_telegram_message(message):
    """Send a message to Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"Telegram message sent: {message}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Telegram message: {e}")

def find_all_prediction_files():
    """Find all prediction data files available."""
    print("Looking for all available prediction data files...")
    
    # Path to the prediction_data directory
    prediction_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "prediction_data")
    
    # Look for all JSON files
    all_files = glob.glob(os.path.join(prediction_data_dir, "*.json"))
    
    if not all_files:
        error_msg = "No prediction data files found in the prediction_data directory."
        print(error_msg)
        send_telegram_message(f"❌ {error_msg}")
        return []
    
    # Sort files by date (oldest first)
    all_files.sort(key=os.path.getmtime)
    
    print(f"Found {len(all_files)} prediction files.")
    return all_files

def load_historical_price_data():
    """Load all historical price data from the CSV file."""
    print("Loading historical price data...")
    
    # Path to the historical data file
    data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "Data", "CSV", "dailymergedv3.csv")
    
    # Check if the file exists
    if not os.path.exists(data_file):
        print(f"Error: Historical data file not found at {data_file}")
        return None
    
    try:
        # Load the historical data
        print(f"Loading historical data from {data_file}...")
        data = pd.read_csv(data_file)
        
        # Convert date columns if needed
        date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y']
        for date_format in date_formats:
            try:
                sample = data['DataUpdateDate'].iloc[0] if not data.empty else ""
                pd.to_datetime(sample, format=date_format)
                print(f"Detected date format: {date_format}")
                data['DataUpdateDate'] = pd.to_datetime(data['DataUpdateDate'], format=date_format, errors='coerce')
                break
            except:
                continue
        
        # Drop rows with missing DataUpdateDate
        data = data.dropna(subset=['DataUpdateDate'])
        
        # Sort by date
        data = data.sort_values(by='DataUpdateDate')
        
        # Get date range
        min_date = data['DataUpdateDate'].min()
        max_date = data['DataUpdateDate'].max()
        print(f"Historical data available from {min_date} to {max_date}")
        print(f"Total of {len(data)} price records for {data['name'].nunique()} cryptocurrencies")
        
        return data
    
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return None

def extract_date_from_filename(filename):
    """Extract the date from a prediction file name."""
    try:
        # Extract date from filename (format: crypto_predictions_batch{batch_num}_{date}_{time}.json)
        file_name = os.path.basename(filename)
        date_parts = file_name.split('_')
        
        if len(date_parts) >= 4:
            date_str = date_parts[3].split('.')[0][:8]  # Extract YYYYMMDD part
            return datetime.strptime(date_str, '%Y%m%d').date()
    except Exception as e:
        print(f"Error extracting date from filename {filename}: {e}")
    
    # If we can't extract the date, use the file modification time
    return datetime.fromtimestamp(os.path.getmtime(filename)).date()

def score_prediction(predicted_price, actual_price):
    """
    Score a prediction based on accuracy and direction:
    - 3 points: Perfect match (predicted price is within ±2% of actual price)
    - 2 points: Conservative win (actual price > predicted price by more than 2%)
    - 1 point: Close miss (actual price < predicted price, but within 5%)
    - 0 points: Significant miss (actual price < predicted price by more than 5%)
    
    This scoring system rewards:
    1. Accurate predictions (close matches)
    2. Conservative predictions (where actual performance exceeds prediction)
    3. Slightly penalizes optimistic predictions that don't materialize
    """
    if predicted_price <= 0 or actual_price <= 0:
        return 0
    
    # Calculate percentage difference
    percentage_diff = abs(((predicted_price - actual_price) / actual_price) * 100)
    
    # Calculate the direction of the error
    # positive_error means predicted price was higher than actual (optimistic prediction)
    # negative_error means predicted price was lower than actual (conservative prediction)
    positive_error = predicted_price > actual_price
    
    # Perfect match: within 2% regardless of direction
    if percentage_diff <= 2:
        return 3
    
    # Conservative win: actual price exceeded prediction by more than 2%
    if not positive_error and percentage_diff > 2:
        return 2
    
    # Close miss: optimistic prediction but within 5%
    if positive_error and percentage_diff <= 5:
        return 1
    
    # Significant miss: optimistic prediction off by more than 5%
    return 0

def find_actual_price(historical_data, crypto_name, prediction_date, target_date):
    """Find the actual price for a cryptocurrency on a specific date."""
    try:
        # Convert prediction_date string to datetime
        if isinstance(prediction_date, str):
            pred_date = datetime.strptime(prediction_date.split()[0], '%Y-%m-%d').date()
        else:
            pred_date = prediction_date
        
        # Calculate the target date (when we want to check the actual price)
        if isinstance(target_date, str):
            actual_date = datetime.strptime(target_date.split()[0], '%Y-%m-%d').date()
        else:
            actual_date = target_date
        
        # Filter historical data for this cryptocurrency
        crypto_data = historical_data[historical_data['name'] == crypto_name].copy()
        
        if crypto_data.empty:
            return None
        
        # Extract date from DataUpdateDate for comparison
        crypto_data.loc[:, 'date'] = crypto_data['DataUpdateDate'].dt.date
        
        # Calculate the absolute difference between each date and the target date
        crypto_data.loc[:, 'date_diff'] = crypto_data['date'].apply(lambda x: abs((x - actual_date).days))
        
        # Sort by date difference and get the closest date
        crypto_data = crypto_data.sort_values('date_diff')
        
        # Get the closest date entry
        if not crypto_data.empty:
            closest_entry = crypto_data.iloc[0]
            date_diff = closest_entry['date_diff']
            
            # Only use the price if it's within 3 days of the target date
            if date_diff <= 3:
                return {
                    'price': float(closest_entry['current_price']),
                    'date': closest_entry['DataUpdateDate'].strftime('%Y-%m-%d')
                }
        
        return None
    
    except Exception as e:
        print(f"Error finding actual price for {crypto_name} on {target_date}: {e}")
        return None

def backtest_predictions_historical(prediction_files, historical_data):
    """Backtest predictions against historical price data."""
    print(f"Backtesting predictions from {len(prediction_files)} files...")
    
    # Dictionary to store all backtesting results
    all_results = []
    
    # Get list of all cryptocurrencies in historical data
    available_cryptos = set(historical_data['name'].unique())
    print(f"Found {len(available_cryptos)} cryptocurrencies in historical data")
    
    # Get the most recent date in the historical data
    most_recent_date = historical_data['DataUpdateDate'].max().date()
    print(f"Most recent date in historical data: {most_recent_date}")
    
    # Limit the number of files to process to avoid memory issues
    max_files = 50  # Increased to process more files
    if len(prediction_files) > max_files:
        print(f"Limiting to {max_files} most recent files to avoid memory issues")
        prediction_files = sorted(prediction_files, key=os.path.getmtime, reverse=True)[:max_files]
    
    # Debug counters
    total_cryptos_processed = 0
    cryptos_not_found = 0
    future_dates_skipped = 0
    no_actual_price_found = 0
    successful_backtests = 0
    
    # Process each prediction file
    for prediction_file in prediction_files:
        print(f"Processing {os.path.basename(prediction_file)}...")
        
        # Extract the date from the filename
        file_date = extract_date_from_filename(prediction_file)
        
        try:
            with open(prediction_file, 'r') as f:
                predictions = json.load(f)
            
            print(f"File contains predictions for {len(predictions)} cryptocurrencies")
            
            # Process each cryptocurrency in the file
            for crypto_name, prediction_data in predictions.items():
                total_cryptos_processed += 1
                
                # Check if cryptocurrency exists in historical data
                if crypto_name not in available_cryptos:
                    cryptos_not_found += 1
                    if cryptos_not_found <= 10:  # Limit logging to avoid spam
                        print(f"Cryptocurrency not found in historical data: {crypto_name}")
                    continue
                
                # Check if we have prediction data
                if prediction_data:
                    # Filter prediction data to only include past dates
                    past_predictions = []
                    for day_prediction in prediction_data:
                        try:
                            prediction_date = day_prediction['DataUpdateDate']
                            pred_date = datetime.strptime(prediction_date.split()[0], '%Y-%m-%d').date()
                            
                            # Only include predictions from the past
                            if pred_date < most_recent_date:
                                past_predictions.append(day_prediction)
                        except Exception as e:
                            print(f"Error parsing prediction date '{day_prediction.get('DataUpdateDate', 'unknown')}': {e}")
                    
                    if not past_predictions:
                        continue
                    
                    print(f"Found {len(past_predictions)} past predictions for {crypto_name}")
                    
                    # Process each past prediction
                    for day_prediction in past_predictions:
                        # Get the prediction details
                        prediction_date = day_prediction['DataUpdateDate']
                        predicted_price = float(day_prediction['predicted_price'])
                        
                        # Parse the prediction date
                        pred_date = datetime.strptime(prediction_date.split()[0], '%Y-%m-%d').date()
                        
                        # For each prediction, we'll check the actual price at different time intervals
                        # Limit to fewer time horizons to reduce processing time
                        for days_later in [1, 7, 30]:
                            # Calculate the target date
                            target_date = pred_date + timedelta(days=days_later)
                            
                            # Skip if the target date is beyond the most recent date in our historical data
                            if target_date > most_recent_date:
                                future_dates_skipped += 1
                                if future_dates_skipped <= 5:  # Limit logging
                                    print(f"Skipping future date: {target_date} (prediction from {pred_date}, most recent data: {most_recent_date})")
                                continue
                            
                            # Find the actual price on the target date
                            actual_price_data = find_actual_price(historical_data, crypto_name, pred_date, target_date)
                            
                            if actual_price_data:
                                actual_price = actual_price_data['price']
                                actual_date = actual_price_data['date']
                                
                                # Score the prediction
                                score = score_prediction(predicted_price, actual_price)
                                
                                # Calculate percentage change
                                if predicted_price > 0 and actual_price > 0:
                                    percentage_change = ((actual_price - predicted_price) / predicted_price) * 100
                                    percentage_diff = abs(percentage_change)
                                else:
                                    percentage_change = float('inf')
                                    percentage_diff = float('inf')
                                
                                # Store the result
                                result = {
                                    'crypto_name': crypto_name,
                                    'file_date': file_date.strftime('%Y-%m-%d'),
                                    'prediction_date': prediction_date,
                                    'predicted_price': predicted_price,
                                    'target_date': target_date.strftime('%Y-%m-%d'),
                                    'actual_date': actual_date,
                                    'actual_price': actual_price,
                                    'days_later': days_later,
                                    'score': score,
                                    'percentage_change': percentage_change,
                                    'percentage_diff': percentage_diff
                                }
                                
                                all_results.append(result)
                                successful_backtests += 1
                                
                                # Log some successful results for debugging
                                if successful_backtests <= 5:
                                    print(f"Successful backtest: {crypto_name}, predicted: {predicted_price}, actual: {actual_price}, score: {score}")
                                elif successful_backtests == 6:
                                    print("More successful backtests found...")
                                
                                # Periodically clear memory to avoid issues with large datasets
                                if len(all_results) % 1000 == 0:
                                    print(f"Processed {len(all_results)} results so far...")
                                    import gc
                                    gc.collect()
                            else:
                                no_actual_price_found += 1
                                if no_actual_price_found <= 10:  # Limit logging
                                    print(f"No actual price found for {crypto_name} on {target_date}")
        
        except Exception as e:
            print(f"Error processing {prediction_file}: {e}")
    
    # Print debug summary
    print("\nBacktesting Debug Summary:")
    print(f"Total cryptocurrencies processed: {total_cryptos_processed}")
    print(f"Cryptocurrencies not found in historical data: {cryptos_not_found}")
    print(f"Future dates skipped: {future_dates_skipped}")
    print(f"No actual price found: {no_actual_price_found}")
    print(f"Successful backtests: {successful_backtests}")
    
    print(f"Generated {len(all_results)} backtest results.")
    return all_results

def save_backtest_results(results):
    """Save backtesting results to JSON and CSV files."""
    print("Saving backtesting results...")
    
    if not results:
        print("No results to save.")
        return None, None
    
    # Create directory for backtesting results if it doesn't exist
    backtest_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "backtest_results")
    
    if not os.path.exists(backtest_dir):
        os.makedirs(backtest_dir)
    
    # Current date for the output files
    current_date = datetime.now().strftime('%Y%m%d')
    
    # Save to JSON
    json_filename = os.path.join(backtest_dir, f"historical_backtest_results_{current_date}.json")
    
    with open(json_filename, 'w') as json_file:
        json.dump(results, json_file, indent=2, default=str)
    
    print(f"Saved JSON results to {json_filename}")
    
    # Save to CSV
    csv_filename = os.path.join(backtest_dir, f"historical_backtest_results_{current_date}.csv")
    
    # Get field names from the first result
    fieldnames = results[0].keys()
    
    try:
        # Use UTF-8 encoding to handle special characters in cryptocurrency names
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Saved CSV results to {csv_filename}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        print("Trying alternative approach...")
        
        # If UTF-8 encoding fails, try saving with error handling for problematic characters
        try:
            # Create a sanitized version of the data
            sanitized_results = []
            for item in results:
                sanitized_item = {}
                for key, value in item.items():
                    if isinstance(value, str):
                        # Replace or remove problematic characters
                        sanitized_item[key] = value.encode('ascii', 'replace').decode('ascii')
                    else:
                        sanitized_item[key] = value
                sanitized_results.append(sanitized_item)
            
            with open(csv_filename, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sanitized_results)
            
            print(f"Saved sanitized CSV results to {csv_filename}")
        except Exception as e2:
            print(f"Error saving sanitized CSV file: {e2}")
            print("CSV file could not be saved. JSON file was saved successfully.")
    
    return json_filename, csv_filename

def analyze_backtest_results(results):
    """Analyze backtesting results and generate summary statistics."""
    print("Analyzing backtesting results...")
    
    if not results:
        print("No results to analyze.")
        return None
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Overall statistics
    total_predictions = len(df)
    perfect_match = len(df[df['score'] == 3])
    conservative_win = len(df[df['score'] == 2])
    close_miss = len(df[df['score'] == 1])
    significant_miss = len(df[df['score'] == 0])
    
    perfect_percentage = (perfect_match / total_predictions) * 100 if total_predictions > 0 else 0
    conservative_percentage = (conservative_win / total_predictions) * 100 if total_predictions > 0 else 0
    close_miss_percentage = (close_miss / total_predictions) * 100 if total_predictions > 0 else 0
    significant_miss_percentage = (significant_miss / total_predictions) * 100 if total_predictions > 0 else 0
    
    average_score = df['score'].mean() if not df.empty else 0
    
    # Calculate average percentage difference
    valid_diffs = df[df['percentage_diff'] < float('inf')]['percentage_diff']
    average_diff = valid_diffs.mean() if not valid_diffs.empty else float('inf')
    
    # Analysis by time horizon
    time_horizon_analysis = df.groupby('days_later').agg({
        'score': ['mean', 'count'],
        'percentage_diff': lambda x: x[x < float('inf')].mean()
    }).reset_index()
    
    # Analysis by cryptocurrency
    crypto_performance = df.groupby('crypto_name')['score'].mean().sort_values(ascending=False)
    top_cryptos = crypto_performance.head(20).to_dict()
    
    # Analysis by date
    df['prediction_date_only'] = pd.to_datetime(df['prediction_date'].str.split().str[0])
    date_performance = df.groupby(df['prediction_date_only'].dt.strftime('%Y-%m-%d'))['score'].mean()
    
    # Create summary
    summary = {
        'total_predictions': total_predictions,
        'perfect_match': perfect_match,
        'conservative_win': conservative_win,
        'close_miss': close_miss,
        'significant_miss': significant_miss,
        'perfect_percentage': perfect_percentage,
        'conservative_percentage': conservative_percentage,
        'close_miss_percentage': close_miss_percentage,
        'significant_miss_percentage': significant_miss_percentage,
        'average_score': average_score,
        'average_percentage_diff': average_diff,
        'time_horizon_analysis': time_horizon_analysis.to_dict(),
        'top_performing_cryptos': top_cryptos,
        'date_performance': date_performance.to_dict()
    }
    
    # Generate visualizations
    generate_visualizations(df, backtest_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                                        "backtest_results"))
    
    return summary

def generate_visualizations(df, backtest_dir):
    """Generate visualizations of the backtest results."""
    print("Generating visualizations...")
    
    # Create directory for visualizations if it doesn't exist
    if not os.path.exists(backtest_dir):
        os.makedirs(backtest_dir)
    
    # 1. Score distribution by time horizon
    plt.figure(figsize=(12, 8))
    score_by_horizon = df.groupby('days_later')['score'].mean()
    score_by_horizon.plot(kind='bar', color='skyblue')
    plt.title('Average Prediction Score by Time Horizon')
    plt.xlabel('Days After Prediction')
    plt.ylabel('Average Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(backtest_dir, 'score_by_time_horizon.png'))
    plt.close()
    
    # 2. Top 10 cryptocurrencies by prediction accuracy
    plt.figure(figsize=(14, 8))
    top_cryptos = df.groupby('crypto_name')['score'].mean().sort_values(ascending=False).head(10)
    top_cryptos.plot(kind='bar', color='lightgreen')
    plt.title('Top 10 Cryptocurrencies by Prediction Accuracy')
    plt.xlabel('Cryptocurrency')
    plt.ylabel('Average Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(backtest_dir, 'top_cryptos_by_accuracy.png'))
    plt.close()
    
    # 3. Prediction accuracy over time
    plt.figure(figsize=(15, 8))
    df['prediction_date_only'] = pd.to_datetime(df['prediction_date'].str.split().str[0])
    time_series = df.groupby(df['prediction_date_only'])['score'].mean()
    time_series.plot(marker='o', linestyle='-', markersize=4, color='coral')
    plt.title('Prediction Accuracy Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(backtest_dir, 'accuracy_over_time.png'))
    plt.close()
    
    # 4. Score distribution
    plt.figure(figsize=(10, 6))
    score_counts = df['score'].value_counts().sort_index()
    score_counts.plot(kind='bar', color=['red', 'orange', 'lightgreen', 'darkgreen'])
    plt.title('Distribution of Prediction Scores')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.xticks([0, 1, 2, 3], ['Significant Miss (0)', 'Close Miss (1)', 'Conservative Win (2)', 'Perfect Match (3)'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(backtest_dir, 'score_distribution.png'))
    plt.close()
    
    print(f"Visualizations saved to {backtest_dir}")

def main():
    """Main function to run the historical backtesting process."""
    start_time = datetime.now()
    script_name = os.path.basename(__file__)
    
    # Send initial Telegram notification
    start_message = f"🔍 Starting historical prediction backtesting process ({script_name}) at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    print(start_message)
    send_telegram_message(start_message)
    
    try:
        # Find all prediction files
        prediction_files = find_all_prediction_files()
        
        if not prediction_files:
            error_msg = "No prediction files found for backtesting."
            print(error_msg)
            send_telegram_message(f"❌ {error_msg}")
            return
        
        # Load all historical price data
        historical_data = load_historical_price_data()
        
        if historical_data is None:
            error_msg = "Failed to load historical price data."
            print(error_msg)
            send_telegram_message(f"❌ {error_msg}")
            return
        
        # Backtest predictions against historical data
        backtest_results = backtest_predictions_historical(prediction_files, historical_data)
        
        # Check if we have any results
        if not backtest_results:
            error_msg = "No backtest results were generated. This could be because:\n" + \
                        "1. All prediction dates are in the future\n" + \
                        "2. No matching cryptocurrencies found in historical data\n" + \
                        "3. No actual prices found for the target dates"
            print(error_msg)
            send_telegram_message(f"❌ {error_msg}")
            return
        
        # Save the results
        json_file, csv_file = save_backtest_results(backtest_results)
        
        if json_file is None or csv_file is None:
            error_msg = "Failed to save backtest results."
            print(error_msg)
            send_telegram_message(f"❌ {error_msg}")
            return
        
        # Analyze the results
        analysis = analyze_backtest_results(backtest_results)
        
        if analysis:
            # Print the analysis
            print("\nHistorical Backtesting Analysis:")
            print("=" * 80)
            print(f"Total predictions: {analysis['total_predictions']}")
            print(f"Perfect match (score 3): {analysis['perfect_match']} ({analysis['perfect_percentage']:.2f}%)")
            print(f"Conservative win (score 2): {analysis['conservative_win']} ({analysis['conservative_percentage']:.2f}%)")
            print(f"Close miss (score 1): {analysis['close_miss']} ({analysis['close_miss_percentage']:.2f}%)")
            print(f"Significant miss (score 0): {analysis['significant_miss']} ({analysis['significant_miss_percentage']:.2f}%)")
            print(f"Average score: {analysis['average_score']:.2f}")
            print(f"Average percentage difference: {analysis['average_percentage_diff']:.2f}%")
            
            print("\nPerformance by Time Horizon:")
            for days, stats in analysis['time_horizon_analysis'].items():
                if days == 'days_later':
                    continue
                print(f"{days} days: Average score = {stats['score']['mean']:.2f} ({stats['score']['count']} predictions)")
            
            print("\nTop 10 Performing Cryptocurrencies:")
            for i, (crypto, score) in enumerate(list(analysis['top_performing_cryptos'].items())[:10], 1):
                print(f"{i}. {crypto}: {score:.2f}")
            
            # Send analysis to Telegram
            analysis_msg = (
                f"📊 Historical Backtesting Analysis:\n\n"
                f"Total predictions: {analysis['total_predictions']}\n"
                f"Perfect match (score 3): {analysis['perfect_match']} ({analysis['perfect_percentage']:.2f}%)\n"
                f"Conservative win (score 2): {analysis['conservative_win']} ({analysis['conservative_percentage']:.2f}%)\n"
                f"Close miss (score 1): {analysis['close_miss']} ({analysis['close_miss_percentage']:.2f}%)\n"
                f"Significant miss (score 0): {analysis['significant_miss']} ({analysis['significant_miss_percentage']:.2f}%)\n"
                f"Average score: {analysis['average_score']:.2f}\n"
                f"Average percentage difference: {analysis['average_percentage_diff']:.2f}%\n\n"
                f"Top 5 Performing Cryptocurrencies:\n"
            )
            
            # Add top 5 cryptos to the message
            for i, (crypto, score) in enumerate(list(analysis['top_performing_cryptos'].items())[:5], 1):
                analysis_msg += f"{i}. {crypto}: {score:.2f}\n"
            
            send_telegram_message(analysis_msg)
        
        # Calculate execution time
        end_time = datetime.now()
        duration = end_time - start_time
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Send completion notification
        completion_message = (
            f"✅ Historical backtesting process completed successfully!\n"
            f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Duration: {hours}h {minutes}m {seconds}s\n"
            f"Analyzed {len(prediction_files)} prediction files\n"
            f"Generated {len(backtest_results)} backtest results\n"
            f"Results saved to:\n"
            f"- {os.path.basename(json_file)}\n"
            f"- {os.path.basename(csv_file)}\n"
            f"- Visualizations in backtest_results folder"
        )
        print("\n" + completion_message)
        send_telegram_message(completion_message)
        
    except Exception as e:
        # Send error notification
        error_message = f"❌ Error in {script_name}: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        send_telegram_message(error_message)
        raise

if __name__ == "__main__":
    main()
