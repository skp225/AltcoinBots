#!/usr/bin/env python3
"""
Auto Predict Top Gainers

This script automates the process of:
1. Finding the most recent prediction data files
2. Analyzing the 7-day price predictions for all cryptocurrencies
3. Identifying the top 20 projects most likely to increase in price (percentage)
4. Opening the HTML charts for these projects

The script will:
- Look for today's prediction data files in the "prediction_data" folder
- If today's data is not available, it will use yesterday's files
- Calculate the percentage change over 7 days for each cryptocurrency
- Identify the top 20 projects with the highest predicted percentage increase
- Open the HTML charts for these projects
"""

import os
import sys
import json
import glob
import webbrowser
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
            sys.exit(1)
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

def find_most_recent_prediction_files():
    """Find the most recent prediction data files."""
    print("Looking for the most recent prediction data files...")
    
    # Path to the prediction_data directory
    prediction_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "prediction_data")
    
    # Check if the directory exists
    if not os.path.exists(prediction_data_dir):
        error_msg = f"Prediction data directory not found at {prediction_data_dir}"
        print(error_msg)
        send_telegram_message(f"❌ {error_msg}")
        sys.exit(1)
    
    # Get all JSON files in the prediction_data directory
    all_files = glob.glob(os.path.join(prediction_data_dir, "*.json"))
    
    if not all_files:
        error_msg = "No prediction data files found in the prediction_data directory."
        print(error_msg)
        send_telegram_message(f"❌ {error_msg}")
        sys.exit(1)
    
    # Get the current date
    current_date = datetime.now()
    
    # Format dates for file pattern matching
    today_str = current_date.strftime('%Y%m%d')
    yesterday_str = (current_date - timedelta(days=1)).strftime('%Y%m%d')
    day_before_yesterday_str = (current_date - timedelta(days=2)).strftime('%Y%m%d')
    
    # Dictionary to store files by date
    files_by_date = {}
    
    # Process each file to extract its date
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        file_date = None
        
        # Try to extract date from filename using different patterns
        # Pattern 1: crypto_predictions_batch{batch_num}_{date}_{time}.json
        parts = file_name.split('_')
        if len(parts) >= 4 and parts[0] == "crypto" and parts[1] == "predictions":
            try:
                # Try to extract date from the 4th part (index 3)
                date_part = parts[3]
                if len(date_part) >= 8 and date_part.isdigit():
                    file_date = date_part[:8]  # Extract YYYYMMDD part
            except (IndexError, ValueError):
                pass
        
        # If date extraction failed, try using file modification time
        if not file_date:
            try:
                mod_time = os.path.getmtime(file_path)
                mod_date = datetime.fromtimestamp(mod_time)
                file_date = mod_date.strftime('%Y%m%d')
            except Exception as e:
                print(f"Error getting modification time for {file_name}: {e}")
                continue
        
        # Add file to the appropriate date bucket
        if file_date not in files_by_date:
            files_by_date[file_date] = []
        files_by_date[file_date].append(file_path)
    
    # Print found dates for debugging
    print(f"Found files for these dates: {sorted(files_by_date.keys(), reverse=True)}")
    
    # Try to find files for today, yesterday, or day before yesterday
    for date_str in [today_str, yesterday_str, day_before_yesterday_str]:
        if date_str in files_by_date and files_by_date[date_str]:
            print(f"Found {len(files_by_date[date_str])} prediction files for {date_str}.")
            if date_str == today_str:
                send_telegram_message(f"📊 Using today's prediction data ({date_str})")
            elif date_str == yesterday_str:
                send_telegram_message(f"📊 Using yesterday's prediction data ({date_str})")
            else:
                send_telegram_message(f"📊 Using prediction data from {date_str}")
            return files_by_date[date_str], date_str
    
    # If no files found for the last three days, use the most recent date
    if files_by_date:
        # Sort dates in descending order (newest first)
        sorted_dates = sorted(files_by_date.keys(), reverse=True)
        most_recent_date = sorted_dates[0]
        print(f"Using the most recent prediction data files from {most_recent_date}.")
        send_telegram_message(f"📊 Using the most recent prediction data from {most_recent_date}")
        return files_by_date[most_recent_date], most_recent_date
    
    # Fallback: sort by modification time if date extraction failed completely
    print("Date extraction failed. Sorting files by modification time...")
    all_files.sort(key=os.path.getmtime, reverse=True)
    send_telegram_message("📊 Using the most recent prediction data files (sorted by modification time)")
    return all_files[:10], "recent"  # Return up to 10 most recent files

def analyze_prediction_data(json_files):
    """Analyze the prediction data to find the top gainers."""
    print(f"Analyzing prediction data from {len(json_files)} files...")
    
    # Dictionary to store all cryptocurrency predictions
    all_crypto_predictions = {}
    
    # Process each JSON file
    for json_file in json_files:
        print(f"Processing {os.path.basename(json_file)}...")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Process each cryptocurrency in the file
            for crypto_name, predictions in data.items():
                if crypto_name not in all_crypto_predictions:
                    all_crypto_predictions[crypto_name] = predictions
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"Found predictions for {len(all_crypto_predictions)} cryptocurrencies.")
    
    # Calculate percentage change for each cryptocurrency
    crypto_gains = []
    
    for crypto_name, predictions in all_crypto_predictions.items():
        if len(predictions) < 2:
            print(f"Skipping {crypto_name} due to insufficient prediction data.")
            continue
        
        try:
            # Get the first and last day predictions
            first_day_price = float(predictions[0]['predicted_price'])
            last_day_price = float(predictions[-1]['predicted_price'])
            
            # Calculate percentage change
            if first_day_price > 0:
                percentage_change = ((last_day_price - first_day_price) / first_day_price) * 100
                
                # Store the result
                crypto_gains.append({
                    'name': crypto_name,
                    'first_day_price': first_day_price,
                    'last_day_price': last_day_price,
                    'percentage_change': percentage_change
                })
            else:
                print(f"Skipping {crypto_name} due to zero or negative first day price.")
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error calculating gains for {crypto_name}: {e}")
            continue
    
    # Sort by percentage change (descending)
    crypto_gains.sort(key=lambda x: x['percentage_change'], reverse=True)
    
    return crypto_gains

def find_html_charts(date_str, batch_numbers):
    """Find the HTML chart files for the specified date and batch numbers."""
    print(f"Looking for HTML chart files for date {date_str} and batches {batch_numbers}...")
    
    # Path to the parent directory (AltCoinResearch)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Dictionary to store batch -> html file mapping
    batch_html_files = {}
    
    # Look for HTML files for each batch
    for batch_num in batch_numbers:
        # Pattern for HTML files
        html_pattern = os.path.join(parent_dir, f"crypto_predictions_batch{batch_num}_{date_str}_*.html")
        html_files = glob.glob(html_pattern)
        
        if html_files:
            # Sort by modification time (most recent first)
            html_files.sort(key=os.path.getmtime, reverse=True)
            batch_html_files[batch_num] = html_files[0]
        else:
            # If no files found with exact date, try a more general search
            html_pattern = os.path.join(parent_dir, f"crypto_predictions_batch{batch_num}_*.html")
            html_files = glob.glob(html_pattern)
            
            if html_files:
                # Sort by modification time (most recent first)
                html_files.sort(key=os.path.getmtime, reverse=True)
                batch_html_files[batch_num] = html_files[0]
                print(f"Found HTML file for batch {batch_num}: {os.path.basename(html_files[0])}")
    
    return batch_html_files

def extract_batch_number(json_file):
    """Extract the batch number from a JSON file name."""
    file_name = os.path.basename(json_file)
    parts = file_name.split('_')
    
    if len(parts) >= 3 and parts[0] == "crypto" and parts[1] == "predictions" and parts[2].startswith("batch"):
        batch_num = parts[2].replace("batch", "")
        try:
            return int(batch_num)
        except ValueError:
            return None
    
    return None

def open_html_charts(top_gainers, date_str, json_files):
    """Open HTML charts for the top gaining cryptocurrencies."""
    # Extract batch numbers from JSON files
    batch_numbers = set()
    for json_file in json_files:
        batch_num = extract_batch_number(json_file)
        if batch_num is not None:
            batch_numbers.add(batch_num)
    
    print(f"Found batch numbers: {batch_numbers}")
    
    # Find HTML chart files
    batch_html_files = find_html_charts(date_str, batch_numbers)
    
    if not batch_html_files:
        error_msg = f"No HTML chart files found for date {date_str}."
        print(error_msg)
        send_telegram_message(f"❌ {error_msg}")
        return
    
    print(f"Found HTML files for batches: {list(batch_html_files.keys())}")
    
    # Create a mapping of cryptocurrency name to HTML file
    crypto_html_mapping = {}
    
    # For each batch, check if any of the top gainers are in that batch
    for batch_num, html_file in batch_html_files.items():
        # Find the corresponding JSON file
        json_file = None
        for jf in json_files:
            if extract_batch_number(jf) == batch_num:
                json_file = jf
                break
        
        if json_file:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Check if any of the top gainers are in this batch
                for gainer in top_gainers:
                    if gainer['name'] in data:
                        crypto_html_mapping[gainer['name']] = html_file
                        print(f"Mapped {gainer['name']} to {os.path.basename(html_file)}")
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    # Open HTML charts for the top gainers
    opened_charts = 0
    
    for gainer in top_gainers:
        if gainer['name'] in crypto_html_mapping:
            html_file = crypto_html_mapping[gainer['name']]
            print(f"Opening chart for {gainer['name']} (expected gain: {gainer['percentage_change']:.2f}%)")
            
            try:
                # Ensure the path is absolute and properly formatted for Windows
                abs_path = os.path.abspath(html_file)
                url = f"file:///{abs_path.replace(os.sep, '/')}"
                print(f"Opening URL: {url}")
                
                # Open the HTML file in the default web browser
                webbrowser.open(url)
                opened_charts += 1
                
                # Add a small delay to prevent overwhelming the system
                time.sleep(1)
            except Exception as e:
                print(f"Error opening chart for {gainer['name']}: {e}")
    
    print(f"Opened {opened_charts} charts for the top gainers.")
    send_telegram_message(f"📈 Opened {opened_charts} charts for the top gainers")

def main():
    """Main function to run the top gainers prediction process."""
    start_time = datetime.now()
    script_name = os.path.basename(__file__)
    
    # Send initial Telegram notification
    start_message = f"🚀 Starting top gainers prediction process ({script_name}) at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    print(start_message)
    send_telegram_message(start_message)
    
    try:
        # Find the most recent prediction data files
        json_files, date_str = find_most_recent_prediction_files()
        
        # Analyze the prediction data to find the top gainers
        crypto_gains = analyze_prediction_data(json_files)
        
        # Get the top 20 gainers
        top_gainers = crypto_gains[:20]
        
        # Print the top gainers
        print("\nTop 20 Cryptocurrencies by Predicted 7-Day Gain:")
        print("=" * 80)
        print(f"{'Rank':<5}{'Name':<30}{'First Day':<15}{'Last Day':<15}{'Change (%)':<10}")
        print("-" * 80)
        
        for i, gainer in enumerate(top_gainers, 1):
            print(f"{i:<5}{gainer['name']:<30}{gainer['first_day_price']:<15.8f}{gainer['last_day_price']:<15.8f}{gainer['percentage_change']:<10.2f}")
        
        # Send notification about top gainers
        top_gainers_msg = "🏆 Top 20 Predicted Gainers (7-day):\n"
        for i, gainer in enumerate(top_gainers, 1):
            top_gainers_msg += f"{i}. {gainer['name']}: {gainer['percentage_change']:.2f}%\n"
        send_telegram_message(top_gainers_msg)
        
        # Open HTML charts for the top gainers
        open_html_charts(top_gainers, date_str, json_files)
        
        # Calculate execution time
        end_time = datetime.now()
        duration = end_time - start_time
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Send completion notification
        completion_message = (
            f"✅ Top gainers prediction process completed successfully!\n"
            f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Duration: {hours}h {minutes}m {seconds}s\n"
            f"Analyzed {len(crypto_gains)} cryptocurrencies\n"
            f"Identified top {len(top_gainers)} gainers"
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
