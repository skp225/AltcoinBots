#!/usr/bin/env python3
"""
Backtest Predictions

This script automates the process of:
1. Finding historical prediction data files
2. Getting current/actual price data for the cryptocurrencies
3. Comparing the predictions with the actual prices
4. Scoring the predictions based on accuracy
5. Storing the results in JSON and CSV formats

The script will:
- Look for prediction data files from previous days
- Get current price data for the cryptocurrencies in those predictions
- Score each prediction (2 points if actual price crossed predicted price,
  1 point if within 10%, 0 points otherwise)
- Store the backtesting results as both JSON and CSV files
- Run daily to continuously evaluate prediction accuracy
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
        else:
            print(f"Dependency installer not found at {installer_path}")
            print("Please install required packages manually: pandas, requests")
            subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "requests"], check=True)
            import pandas as pd
            import requests
    except Exception as install_error:
        print(f"Failed to install dependencies: {install_error}")
        print("Please install required packages manually: pandas, requests")
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

def find_prediction_files_to_backtest(days_ago=1):
    """Find prediction data files from a specific number of days ago."""
    print(f"Looking for prediction data files from {days_ago} days ago...")
    
    # Get the date for the files we want to analyze
    target_date = datetime.now() - timedelta(days=days_ago)
    target_date_str = target_date.strftime('%Y%m%d')
    
    # Path to the prediction_data directory
    prediction_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "prediction_data")
    
    # Look for files from the target date
    target_files = glob.glob(os.path.join(prediction_data_dir, f"*_{target_date_str}_*.json"))
    
    if target_files:
        print(f"Found {len(target_files)} prediction files from {target_date_str}.")
        return target_files, target_date_str
    else:
        print(f"No prediction files found for {target_date_str}.")
        
        # Try to find the most recent files before the target date
        all_files = glob.glob(os.path.join(prediction_data_dir, "*.json"))
        
        if not all_files:
            error_msg = "No prediction data files found in the prediction_data directory."
            print(error_msg)
            send_telegram_message(f"❌ {error_msg}")
            return [], None
        
        # Sort files by modification time (most recent first)
        all_files.sort(key=os.path.getmtime, reverse=True)
        
        # Filter files to only include those older than the target date
        filtered_files = []
        filtered_date_str = None
        
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            date_parts = file_name.split('_')
            
            if len(date_parts) >= 4:
                file_date_str = date_parts[3].split('.')[0][:8]  # Extract YYYYMMDD part
                file_date = datetime.strptime(file_date_str, '%Y%m%d')
                
                # Check if this file is older than our target date but not too old
                if file_date < target_date and file_date >= (target_date - timedelta(days=7)):
                    if filtered_date_str is None or file_date_str == filtered_date_str:
                        filtered_date_str = file_date_str
                        filtered_files.append(file_path)
        
        if filtered_files:
            print(f"Found {len(filtered_files)} prediction files from {filtered_date_str} to backtest.")
            return filtered_files, filtered_date_str
        
        print("No suitable prediction files found for backtesting.")
        return [], None

def get_current_crypto_prices(crypto_names):
    """Get current prices for a list of cryptocurrencies from the local historical data file."""
    print(f"Getting current prices for {len(crypto_names)} cryptocurrencies from local data...")
    
    # Path to the historical data file
    data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "Data", "CSV", "dailymergedv3.csv")
    
    # Check if the file exists
    if not os.path.exists(data_file):
        print(f"Error: Historical data file not found at {data_file}")
        return {}
    
    # Dictionary to store crypto prices
    crypto_prices = {}
    
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
        
        # Sort by date to get the most recent data
        data = data.sort_values(by='DataUpdateDate', ascending=False)
        
        # Get the most recent date
        most_recent_date = data['DataUpdateDate'].max()
        print(f"Most recent data date: {most_recent_date}")
        
        # Filter to only include the most recent data
        most_recent_data = data[data['DataUpdateDate'] == most_recent_date]
        
        # Create a mapping of cryptocurrency names to their current prices
        for crypto_name in crypto_names:
            # Find the cryptocurrency in the data
            crypto_data = most_recent_data[most_recent_data['name'] == crypto_name]
            
            if not crypto_data.empty:
                # Get the current price
                current_price = crypto_data['current_price'].iloc[0]
                
                if pd.notna(current_price):
                    crypto_prices[crypto_name] = float(current_price)
                    print(f"Found price for {crypto_name}: ${current_price}")
                else:
                    print(f"No price data for {crypto_name}")
            else:
                print(f"Cryptocurrency {crypto_name} not found in historical data")
        
    except Exception as e:
        print(f"Error processing historical data: {e}")
    
    print(f"Successfully retrieved prices for {len(crypto_prices)} out of {len(crypto_names)} cryptocurrencies.")
    return crypto_prices

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

def backtest_predictions(prediction_files, current_prices):
    """Backtest predictions against current prices."""
    print(f"Backtesting predictions from {len(prediction_files)} files...")
    
    # Dictionary to store all backtesting results
    all_results = {}
    
    # Get today's date
    today = datetime.now().date()
    
    # Process each prediction file
    for prediction_file in prediction_files:
        print(f"Processing {os.path.basename(prediction_file)}...")
        
        try:
            with open(prediction_file, 'r') as f:
                predictions = json.load(f)
            
            # Process each cryptocurrency in the file
            for crypto_name, prediction_data in predictions.items():
                if crypto_name in current_prices:
                    actual_price = current_prices[crypto_name]
                    
                    # Check if we have prediction data
                    if prediction_data:
                        # Find the prediction for today (or closest to today)
                        # Sort predictions by date
                        sorted_predictions = sorted(
                            prediction_data, 
                            key=lambda x: abs((datetime.strptime(x['DataUpdateDate'].split()[0], '%Y-%m-%d').date() - today).days)
                        )
                        
                        # Get the prediction closest to today
                        closest_prediction = sorted_predictions[0]
                        predicted_price = float(closest_prediction['predicted_price'])
                        prediction_date = closest_prediction['DataUpdateDate']
                        
                        # Score the prediction
                        score = score_prediction(predicted_price, actual_price)
                        
                        # Store the result
                        result = {
                            'crypto_name': crypto_name,
                            'prediction_date': prediction_date,
                            'predicted_price': predicted_price,
                            'actual_price': actual_price,
                            'score': score,
                            'percentage_diff': abs(((predicted_price - actual_price) / actual_price) * 100) if actual_price > 0 else float('inf')
                        }
                        
                        # Store the result for this cryptocurrency
                        if crypto_name not in all_results:
                            all_results[crypto_name] = [result]
                        else:
                            # If we already have a result for this crypto, keep the one with the better score
                            existing_score = all_results[crypto_name][0]['score']
                            
                            if score > existing_score:
                                all_results[crypto_name] = [result]
                else:
                    print(f"No current price data for {crypto_name}, skipping...")
        
        except Exception as e:
            print(f"Error processing {prediction_file}: {e}")
    
    return all_results

def save_backtest_results(results, date_str):
    """Save backtesting results to JSON and CSV files."""
    print("Saving backtesting results...")
    
    # Create directory for backtesting results if it doesn't exist
    backtest_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "backtest_results")
    
    if not os.path.exists(backtest_dir):
        os.makedirs(backtest_dir)
    
    # Current date for the output files
    current_date = datetime.now().strftime('%Y%m%d')
    
    # Prepare data for CSV
    csv_data = []
    
    for crypto_name, crypto_results in results.items():
        for result in crypto_results:
            csv_data.append(result)
    
    # Save to JSON
    json_filename = os.path.join(backtest_dir, f"backtest_results_{date_str}_on_{current_date}.json")
    
    with open(json_filename, 'w') as json_file:
        json.dump(results, json_file, indent=2)
    
    print(f"Saved JSON results to {json_filename}")
    
    # Save to CSV
    csv_filename = os.path.join(backtest_dir, f"backtest_results_{date_str}_on_{current_date}.csv")
    
    if csv_data:
        # Get field names from the first result
        fieldnames = csv_data[0].keys()
        
        try:
            # Use UTF-8 encoding to handle special characters in cryptocurrency names
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"Saved CSV results to {csv_filename}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")
            print("Trying alternative approach...")
            
            # If UTF-8 encoding fails, try saving with error handling for problematic characters
            try:
                # Create a sanitized version of the data
                sanitized_data = []
                for item in csv_data:
                    sanitized_item = {}
                    for key, value in item.items():
                        if isinstance(value, str):
                            # Replace or remove problematic characters
                            sanitized_item[key] = value.encode('ascii', 'replace').decode('ascii')
                        else:
                            sanitized_item[key] = value
                    sanitized_data.append(sanitized_item)
                
                with open(csv_filename, 'w', newline='') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(sanitized_data)
                
                print(f"Saved sanitized CSV results to {csv_filename}")
            except Exception as e2:
                print(f"Error saving sanitized CSV file: {e2}")
                print("CSV file could not be saved. JSON file was saved successfully.")
    else:
        print("No data to save to CSV.")
    
    return json_filename, csv_filename

def analyze_backtest_results(results):
    """Analyze backtesting results and generate summary statistics."""
    print("Analyzing backtesting results...")
    
    if not results:
        print("No results to analyze.")
        return None
    
    # Flatten the results for analysis
    flat_results = []
    for crypto_name, crypto_results in results.items():
        for result in crypto_results:
            flat_results.append(result)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(flat_results)
    
    # Calculate summary statistics
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
    
    # Get top performing cryptocurrencies
    crypto_performance = df.groupby('crypto_name')['score'].mean().sort_values(ascending=False)
    top_cryptos = crypto_performance.head(10).to_dict()
    
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
        'top_performing_cryptos': top_cryptos
    }
    
    return summary

def main():
    """Main function to run the backtesting process."""
    start_time = datetime.now()
    script_name = os.path.basename(__file__)
    
    # Send initial Telegram notification
    start_message = f"🔍 Starting prediction backtesting process ({script_name}) at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    print(start_message)
    send_telegram_message(start_message)
    
    try:
        # Find prediction files to backtest (default: from yesterday)
        prediction_files, date_str = find_prediction_files_to_backtest(days_ago=1)
        
        if not prediction_files:
            error_msg = "No prediction files found for backtesting."
            print(error_msg)
            send_telegram_message(f"❌ {error_msg}")
            return
        
        # Extract all cryptocurrency names from the prediction files
        crypto_names = set()
        
        for prediction_file in prediction_files:
            try:
                with open(prediction_file, 'r') as f:
                    predictions = json.load(f)
                
                for crypto_name in predictions.keys():
                    crypto_names.add(crypto_name)
            except Exception as e:
                print(f"Error extracting crypto names from {prediction_file}: {e}")
        
        print(f"Found {len(crypto_names)} unique cryptocurrencies in the prediction files.")
        
        # Get current prices for the cryptocurrencies from local historical data
        current_prices = get_current_crypto_prices(list(crypto_names))
        
        # Backtest the predictions
        backtest_results = backtest_predictions(prediction_files, current_prices)
        
        # Save the results
        json_file, csv_file = save_backtest_results(backtest_results, date_str)
        
        # Analyze the results
        analysis = analyze_backtest_results(backtest_results)
        
        if analysis:
            # Print the analysis
            print("\nBacktesting Analysis:")
            print("=" * 80)
            print(f"Total predictions: {analysis['total_predictions']}")
            print(f"Perfect match (score 3): {analysis['perfect_match']} ({analysis['perfect_percentage']:.2f}%)")
            print(f"Conservative win (score 2): {analysis['conservative_win']} ({analysis['conservative_percentage']:.2f}%)")
            print(f"Close miss (score 1): {analysis['close_miss']} ({analysis['close_miss_percentage']:.2f}%)")
            print(f"Significant miss (score 0): {analysis['significant_miss']} ({analysis['significant_miss_percentage']:.2f}%)")
            print(f"Average score: {analysis['average_score']:.2f}")
            print(f"Average percentage difference: {analysis['average_percentage_diff']:.2f}%")
            
            print("\nTop 10 Performing Cryptocurrencies:")
            for crypto, score in analysis['top_performing_cryptos'].items():
                print(f"{crypto}: {score:.2f}")
            
            # Send analysis to Telegram
            analysis_msg = (
                f"📊 Backtesting Analysis for {date_str} predictions:\n\n"
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
            f"✅ Backtesting process completed successfully!\n"
            f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Duration: {hours}h {minutes}m {seconds}s\n"
            f"Analyzed {len(crypto_names)} cryptocurrencies\n"
            f"Results saved to:\n"
            f"- {os.path.basename(json_file)}\n"
            f"- {os.path.basename(csv_file)}"
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
