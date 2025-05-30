#!/usr/bin/env python3
"""
Auto Predict Trending Projects

This script automates the process of:
1. Running trend analysis to identify trending projects
2. Feeding those projects to the crypto prediction model
3. Generating prediction charts and machine-readable data

The script will:
- Run trend_analysis_csv.py to generate trending project lists
- Read the 3-day trending projects from the Excel output
- Feed these projects to the crypto prediction model in batches of 10
- Generate prediction charts and data for all trending projects
"""

import os
import sys
import subprocess
import importlib.util
from datetime import datetime, timedelta
import time
import traceback
import glob

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
            print("Please install required packages manually: pandas, requests, openpyxl")
            sys.exit(1)
    except Exception as install_error:
        print(f"Failed to install dependencies: {install_error}")
        print("Please install required packages manually: pandas, requests, openpyxl")
        sys.exit(1)

# Try to ensure openpyxl is installed for Excel file operations
try:
    import openpyxl
except ImportError:
    print("Missing openpyxl package for Excel operations. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
        import openpyxl
        print("Successfully installed openpyxl")
    except Exception as e:
        print(f"Failed to install openpyxl: {e}")
        print("Excel operations may fail. Please install openpyxl manually.")

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

def run_trend_analysis():
    """Find the most recent 3-day trending file or run the analysis to generate a new one."""
    # Path to the Excel output directory
    excel_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "Data", "Excel")
    
    # Check if the Excel directory exists
    if not os.path.exists(excel_dir):
        print(f"Creating Excel directory: {excel_dir}")
        os.makedirs(excel_dir, exist_ok=True)
    
    # First, run the check_and_run_lenient.py script to ensure we have enough trending projects
    print("Running check_and_run_lenient.py to ensure we have enough trending projects...")
    check_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    "check_and_run_lenient.py")
    
    if os.path.exists(check_script_path):
        try:
            # Run with a minimum of 20 projects
            subprocess.run([sys.executable, check_script_path, "20"], check=True)
            print("check_and_run_lenient.py completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: check_and_run_lenient.py failed: {e}")
            # Continue anyway, as we'll fall back to our regular file finding logic
    else:
        print(f"Warning: check_and_run_lenient.py not found at {check_script_path}")
    
    # Get dates for the last few days
    current_date = datetime.now()
    dates = [
        current_date.strftime('%Y%m%d'),
        (current_date - timedelta(days=1)).strftime('%Y%m%d'),
        (current_date - timedelta(days=2)).strftime('%Y%m%d'),
        (current_date - timedelta(days=3)).strftime('%Y%m%d'),
        (current_date - timedelta(days=4)).strftime('%Y%m%d'),
        (current_date - timedelta(days=5)).strftime('%Y%m%d'),
        (current_date - timedelta(days=6)).strftime('%Y%m%d'),
        (current_date - timedelta(days=7)).strftime('%Y%m%d')
    ]
    
    # First, check if today's 3-day trending file exists
    today_file = os.path.join(excel_dir, f"3days_trending_{dates[0]}.xlsx")
    if os.path.exists(today_file):
        print(f"Found existing 3-day trending file for today: {today_file}")
        send_telegram_message(f"📊 Using today's 3-day trending file: {os.path.basename(today_file)}")
        return today_file
    
    # If today's file doesn't exist, try to run the trend analysis
    print("No 3-day trending file found for today. Running trend analysis...")
    send_telegram_message("🔍 Running trend analysis to generate trending projects list...")
    
    # Get the path to the trend_analysis_csv.py script
    trend_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "trend_analysis_csv.py")
    
    # Check if the script exists
    if not os.path.exists(trend_script_path):
        error_msg = f"Error: Trend analysis script not found at {trend_script_path}"
        print(error_msg)
        send_telegram_message(f"❌ {error_msg}")
        sys.exit(1)
    
    # Run the script as a subprocess
    try:
        subprocess.run([sys.executable, trend_script_path], check=True)
        print("Trend analysis completed successfully.")
        send_telegram_message("✅ Trend analysis completed successfully")
    except subprocess.CalledProcessError as e:
        error_msg = f"Error running trend analysis: {e}"
        print(error_msg)
        send_telegram_message(f"❌ {error_msg}")
        # Don't exit here, try to find an existing file instead
    
    # Check if the trend analysis created today's file
    if os.path.exists(today_file):
        print(f"Found newly created 3-day trending file: {today_file}")
        send_telegram_message(f"📊 Using newly created 3-day trending file")
        return today_file
    
    # If today's file still doesn't exist, look for the most recent 3-day trending file
    print("Today's 3-day trending file not found. Looking for recent 3-day trending files...")
    
    # Look for existing 3-day trending files from the last few days
    for date_str in dates[1:]:  # Skip today, already checked
        file_path = os.path.join(excel_dir, f"3days_trending_{date_str}.xlsx")
        if os.path.exists(file_path):
            print(f"Found 3-day trending file from {date_str}: {file_path}")
            send_telegram_message(f"📊 Using 3-day trending file from {date_str}")
            return file_path
    
    # If we still can't find any 3-day trending files, look for any trending files
    print("No 3-day trending files found. Looking for any trending files...")
    
    # Try to find any trending file, prioritizing the most recent ones
    all_trending_files = glob.glob(os.path.join(excel_dir, "*days_trending_*.xlsx"))
    
    if all_trending_files:
        # Sort by modification time (newest first)
        all_trending_files.sort(key=os.path.getmtime, reverse=True)
        newest_file = all_trending_files[0]
        
        print(f"Found trending file: {newest_file}")
        send_telegram_message(f"📊 Using trending file: {os.path.basename(newest_file)}")
        
        return newest_file
    
    # If we still can't find any files, raise an error
    error_msg = f"Error: No trending files found in {excel_dir}"
    print(error_msg)
    send_telegram_message(f"❌ {error_msg}")
    sys.exit(1)

def get_trending_projects(excel_file):
    """Read the trending projects from the Excel file."""
    print(f"Reading trending projects from {excel_file}...")
    
    try:
        df = pd.read_excel(excel_file)
        # Extract project names from the 'name' column
        trending_projects = df['name'].tolist()
        print(f"Found {len(trending_projects)} trending projects.")
        return trending_projects
    except Exception as e:
        error_msg = f"Error reading trending projects: {e}"
        print(error_msg)
        send_telegram_message(f"❌ {error_msg}")
        sys.exit(1)

def import_crypto_predict_model():
    """Import the CryptoPredictModel module."""
    print("Importing CryptoPredictModel module...")
    
    # Path to the CryptoPredictModel.py file
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "Analysis", "CryptoPredictionModel", "CryptoPredictModel.py")
    
    try:
        # Import the module from file path
        spec = importlib.util.spec_from_file_location("CryptoPredictModel", model_path)
        crypto_predict_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crypto_predict_model)
        print("CryptoPredictModel module imported successfully.")
        return crypto_predict_model
    except Exception as e:
        error_msg = f"Error importing CryptoPredictModel: {e}"
        print(error_msg)
        send_telegram_message(f"❌ {error_msg}")
        sys.exit(1)

def load_and_prepare_data(crypto_predict_model):
    """Load and prepare data using functions from CryptoPredictModel."""
    print("Loading and preparing cryptocurrency data...")
    
    # Path to the data file
    data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "Data", "CSV", "dailymergedv3.csv")
    
    # Load the data
    data = pd.read_csv(data_file)
    
    # Clean and preprocess the data (similar to what's done in CryptoPredictModel.main())
    # Convert date columns
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
    
    # Convert other date columns
    data['ath_date'] = pd.to_datetime(data['ath_date'], errors='coerce')
    data['atl_date'] = pd.to_datetime(data['atl_date'], errors='coerce')
    data['last_updated'] = pd.to_datetime(data['last_updated'], errors='coerce')
    
    # Drop rows with missing DataUpdateDate
    data = data.dropna(subset=['DataUpdateDate'])
    
    # Convert numeric columns
    numeric_cols = ['current_price', 'market_cap', 'total_volume', 
                   'price_change_percentage_24h', 'market_cap_change_percentage_24h',
                   'ath_change_percentage', 'atl_change_percentage']
    
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Calculate growth scores
    print("Calculating growth scores...")
    scores = crypto_predict_model.calculate_growth_score(data)
    
    # Get available cryptocurrencies
    available_cryptos = data['name'].unique()
    
    return data, scores, available_cryptos

def process_projects_in_batches(crypto_predict_model, data, scores, available_cryptos, trending_projects, batch_size=10):
    """Process trending projects in batches using the crypto prediction model."""
    print(f"Processing {len(trending_projects)} trending projects in batches of {batch_size}...")
    send_telegram_message(f"🔄 Processing {len(trending_projects)} trending projects in batches of {batch_size}...")
    
    # Store trained models to avoid reprocessing
    trained_models = {}
    
    # Track batch results
    batch_results = []
    
    # Process projects in batches
    for i in range(0, len(trending_projects), batch_size):
        batch = trending_projects[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(trending_projects) + batch_size - 1) // batch_size
        
        print(f"\nProcessing batch {batch_num} of {total_batches}")
        print(f"Projects in this batch: {batch}")
        send_telegram_message(f"🔄 Processing batch {batch_num} of {total_batches} ({len(batch)} projects)")
        
        # Check if projects exist in the dataset
        valid_cryptos = []
        invalid_cryptos = []
        
        for crypto in batch:
            if crypto in available_cryptos:
                valid_cryptos.append(crypto)
            else:
                invalid_cryptos.append(crypto)
        
        if invalid_cryptos:
            print(f"Warning: The following cryptocurrencies were not found in the dataset: {invalid_cryptos}")
            
        if not valid_cryptos:
            print("None of the specified cryptocurrencies were found in the dataset. Skipping this batch.")
            send_telegram_message("⚠️ No valid cryptocurrencies found in this batch. Skipping.")
            continue
        
        # Create tabs for each cryptocurrency
        tabs = []
        
        for crypto_name in valid_cryptos:
            print(f"Processing {crypto_name}...")
            
            # Check if we already have a trained model for this cryptocurrency
            if crypto_name in trained_models:
                print(f"Using cached model for {crypto_name}...")
                model, feature_scaler, target_scaler, features, target, crypto_data = trained_models[crypto_name]
            else:
                # Train a new model
                print(f"Training new model for {crypto_name}...")
                model, feature_scaler, target_scaler, features, target, crypto_data = crypto_predict_model.train_prediction_model(data, crypto_name)
                
                # Store the model if it was successfully trained
                if model is not None:
                    trained_models[crypto_name] = (model, feature_scaler, target_scaler, features, target, crypto_data)
            
            if model is None:
                print(f"Not enough data for {crypto_name}, skipping...")
                continue
                
            # Predict future prices (7 days)
            future_data = crypto_predict_model.predict_future_prices(model, feature_scaler, target_scaler, features, crypto_data, future_days=7)
            
            # Create chart
            chart = crypto_predict_model.create_interactive_chart(crypto_data, future_data, crypto_name)
            
            # Add to tabs
            from bokeh.models import TabPanel
            tabs.append(TabPanel(child=chart, title=crypto_name))
        
        # Create a dictionary to store all prediction data
        all_predictions = {}
        
        # Create the tabs layout
        if tabs:
            from bokeh.models import Tabs
            tabs_layout = Tabs(tabs=tabs)
            
            # Generate a unique filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_output_filename = f"crypto_predictions_batch{batch_num}_{timestamp}.html"
            
            # Create directory for machine-readable outputs if it doesn't exist
            output_dir = "prediction_data"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Prepare machine-readable outputs
            csv_output_filename = os.path.join(output_dir, f"crypto_predictions_batch{batch_num}_{timestamp}.csv")
            json_output_filename = os.path.join(output_dir, f"crypto_predictions_batch{batch_num}_{timestamp}.json")
            
            # Collect all prediction data
            all_predictions_df = pd.DataFrame()
            
            for crypto_name in valid_cryptos:
                if crypto_name in trained_models:
                    model, feature_scaler, target_scaler, features, target, crypto_data = trained_models[crypto_name]
                    
                    if model is not None:
                        # Get the prediction data
                        future_data = crypto_predict_model.predict_future_prices(model, feature_scaler, target_scaler, features, crypto_data, future_days=7)
                        
                        if future_data is not None:
                            # Add cryptocurrency name to the future data
                            future_data['cryptocurrency'] = crypto_name
                            
                            # Add to the combined DataFrame
                            all_predictions_df = pd.concat([all_predictions_df, future_data], ignore_index=True)
                            
                            # Add to the dictionary for JSON output
                            import json
                            crypto_predictions = future_data.to_dict(orient='records')
                            all_predictions[crypto_name] = crypto_predictions
            
            # Save to CSV
            if not all_predictions_df.empty:
                all_predictions_df.to_csv(csv_output_filename, index=False)
                print(f"Prediction data saved to CSV: {csv_output_filename}")
            
            # Save to JSON
            if all_predictions:
                with open(json_output_filename, 'w') as json_file:
                    json.dump(all_predictions, json_file, indent=2, default=str)
                print(f"Prediction data saved to JSON: {json_output_filename}")
            
            # Output to HTML file
            from bokeh.plotting import output_file, show
            output_file(html_output_filename)
            
            # Show the result
            show(tabs_layout)
            print(f"Interactive charts have been generated and saved to '{html_output_filename}'")
            
            # Track batch results
            batch_results.append({
                'batch_num': batch_num,
                'html_file': html_output_filename,
                'csv_file': csv_output_filename,
                'json_file': json_output_filename,
                'projects_count': len(tabs)
            })
            
            # Send Telegram notification for this batch
            batch_msg = (
                f"✅ Batch {batch_num}/{total_batches} completed:\n"
                f"- Processed {len(tabs)} projects\n"
                f"- HTML: {html_output_filename}\n"
                f"- CSV: {os.path.basename(csv_output_filename)}\n"
                f"- JSON: {os.path.basename(json_output_filename)}"
            )
            send_telegram_message(batch_msg)
        else:
            print("No charts were generated for this batch due to insufficient data.")
            send_telegram_message(f"⚠️ Batch {batch_num}/{total_batches}: No charts generated due to insufficient data")
        
        # Add a small delay between batches to allow for file operations to complete
        if i + batch_size < len(trending_projects):
            print("Waiting a few seconds before processing the next batch...")
            time.sleep(5)
    
    return batch_results

def main():
    """Main function to run the automated prediction process."""
    start_time = datetime.now()
    script_name = os.path.basename(__file__)
    
    # Send initial Telegram notification
    start_message = f"🚀 Starting automated prediction process ({script_name}) at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    print(start_message)
    send_telegram_message(start_message)
    
    try:
        # Run trend analysis and get the output file
        three_day_file = run_trend_analysis()
        
        # Get trending projects from the Excel file
        trending_projects = get_trending_projects(three_day_file)
        
        # Send notification about trending projects
        trend_message = f"📊 Found {len(trending_projects)} trending projects from {os.path.basename(three_day_file)}"
        send_telegram_message(trend_message)
        
        # Import the crypto prediction model
        crypto_predict_model = import_crypto_predict_model()
        
        # Load and prepare data
        data, scores, available_cryptos = load_and_prepare_data(crypto_predict_model)
        
        # Process projects in batches
        batch_results = process_projects_in_batches(crypto_predict_model, data, scores, available_cryptos, trending_projects)
        
        # Calculate execution time
        end_time = datetime.now()
        duration = end_time - start_time
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Create summary of results
        total_projects_processed = sum(batch['projects_count'] for batch in batch_results) if batch_results else 0
        
        # Send completion notification
        completion_message = (
            f"✅ Automated prediction process completed successfully!\n"
            f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Duration: {hours}h {minutes}m {seconds}s\n"
            f"Processed {total_projects_processed} of {len(trending_projects)} trending projects\n"
            f"Generated {len(batch_results)} batch files"
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
