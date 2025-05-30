import yfinance as yf
import json
import os
import pandas as pd
import requests
from datetime import datetime, timedelta

# Use dynamic path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Telegram configuration
TELEGRAM_TOKEN = ''
CHAT_ID = ''

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
        # Don't print the actual message content to avoid emoji in console output
        print(f"Telegram message sent successfully")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Telegram message: {e}")

def create_data_folder():
    # Create folder in the same directory as the script
    folder = os.path.join(SCRIPT_DIR, "StockETFData")
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def get_latest_date_from_csv(csv_path):
    """Get the latest date from a CSV file."""
    try:
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        
        # Get the latest date and ensure it's timezone-naive
        latest_date = df.index.max()
        if latest_date.tzinfo is not None:
            latest_date = latest_date.replace(tzinfo=None)
        
        return latest_date
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {str(e)}")
        return None

def get_latest_date_from_json(json_path):
    """Get the latest date from a JSON file."""
    try:
        if not os.path.exists(json_path):
            return None
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data or 'data' not in data or not data['data']:
            return None
        
        # Get the latest date from the keys
        dates = list(data['data'].keys())
        if not dates:
            return None
        
        # Convert to datetime objects for proper comparison (ensure they're timezone-naive)
        date_objects = [datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=None) for date in dates]
        return max(date_objects)
    except Exception as e:
        print(f"Error reading JSON file {json_path}: {str(e)}")
        return None

def update_ticker_data(ticker, data_folder):
    """Update data for a ticker if needed."""
    csv_path = os.path.join(data_folder, f"{ticker}_historical_data.csv")
    json_path = os.path.join(data_folder, f"{ticker}_historical_data.json")
    
    # Get the latest date from both files
    latest_csv_date = get_latest_date_from_csv(csv_path)
    latest_json_date = get_latest_date_from_json(json_path)
    
    # Determine the latest date from both sources
    if latest_csv_date and latest_json_date:
        latest_date = max(latest_csv_date, latest_json_date)
    elif latest_csv_date:
        latest_date = latest_csv_date
    elif latest_json_date:
        latest_date = latest_json_date
    else:
        latest_date = None
    
    # Get today's date
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Check if we need to update
    if latest_date and latest_date.date() >= today.date():
        print(f"Data for {ticker} is already up to date (latest: {latest_date.date()}, today: {today.date()})")
        return
    
    # Determine start date for data download
    if latest_date:
        # Start from the day after the latest date
        start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Updating data for {ticker} from {start_date} to today...")
    else:
        # If no existing data, start from Bitcoin's release date
        start_date = "2009-01-03"
        print(f"No existing data found for {ticker}. Downloading full history from {start_date}...")
    
    end_date = today.strftime("%Y-%m-%d")
    
    try:
        # Download data
        ticker_data = yf.Ticker(ticker)
        hist = ticker_data.history(start=start_date, end=end_date)
        
        if hist.empty:
            print(f"No new data available for {ticker} in the specified date range.")
            return
        
        # Update CSV file
        if os.path.exists(csv_path):
            # Read existing data
            existing_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            # Append new data
            updated_df = pd.concat([existing_df, hist])
            # Remove duplicates (keep the last occurrence)
            updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
            # Sort by date
            updated_df = updated_df.sort_index()
            # Save updated data
            updated_df.to_csv(csv_path)
        else:
            # Save new data
            hist.to_csv(csv_path)
        
        print(f"Updated CSV data for {ticker} in {csv_path}")
        
        # Update JSON file
        if os.path.exists(json_path):
            # Read existing data
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Convert new data to dictionary
            new_data_dict = {}
            for date, row in hist.iterrows():
                date_str = date.strftime("%Y-%m-%d")
                row_dict = row.to_dict()
                new_data_dict[date_str] = row_dict
            
            # Update data dictionary
            json_data['data'].update(new_data_dict)
            json_data['end_date'] = end_date
            json_data['timestamp'] = datetime.now().isoformat()
        else:
            # Convert all data to dictionary
            hist_dict = {}
            for date, row in hist.iterrows():
                date_str = date.strftime("%Y-%m-%d")
                row_dict = row.to_dict()
                hist_dict[date_str] = row_dict
            
            # Create new JSON data
            json_data = {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "timestamp": datetime.now().isoformat(),
                "data": hist_dict
            }
        
        # Save JSON data
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, default=str)
        
        print(f"Updated JSON data for {ticker} in {json_path}")
        print(f"Added {len(hist)} new data points")
        
        if not hist.empty:
            print(f"New data range: {hist.index.min().date()} to {hist.index.max().date()}")
        
    except Exception as e:
        print(f"Failed to update data for {ticker}: {str(e)}")

def main():
    start_time = datetime.now()
    script_name = os.path.basename(__file__)
    
    # Send initial notification with emoji for Telegram but plain text for console
    # Remove emoji from telegram message to avoid encoding issues
    telegram_start_message = f"Starting Stock/ETF daily data update at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    console_start_message = f"Starting Stock/ETF daily data update at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    print(console_start_message)
    send_telegram_message(telegram_start_message)
    
    data_folder = create_data_folder()
    print(f"Data folder: {data_folder}")
    
    crypto_tickers = [
        "BITQ", "SATO", "IBIT", "ARKB", "BITI", "BTF", "SPBC",
        "GBTC", "MARA", "RIOT", "HIVE", "BYON", "MSTR", "SQ",
        "NVDA", "AMD"
    ]
    
    # Track statistics
    updated_tickers = []
    skipped_tickers = []
    error_tickers = []
    
    for ticker in crypto_tickers:
        try:
            # Check if data is already up to date
            csv_path = os.path.join(data_folder, f"{ticker}_historical_data.csv")
            json_path = os.path.join(data_folder, f"{ticker}_historical_data.json")
            
            latest_csv_date = get_latest_date_from_csv(csv_path)
            latest_json_date = get_latest_date_from_json(json_path)
            
            if latest_csv_date and latest_json_date:
                latest_date = max(latest_csv_date, latest_json_date)
            elif latest_csv_date:
                latest_date = latest_csv_date
            elif latest_json_date:
                latest_date = latest_json_date
            else:
                latest_date = None
            
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            if latest_date and latest_date.date() >= today.date():
                skipped_tickers.append(ticker)
            else:
                update_ticker_data(ticker, data_folder)
                updated_tickers.append(ticker)
        except Exception as e:
            error_message = f"Error updating {ticker}: {str(e)}"
            print(error_message)
            error_tickers.append(ticker)
    
    # Calculate execution time
    end_time = datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Create separate messages for console (no emoji) and Telegram (with emoji)
    telegram_completion_message = (
        f"Stock/ETF daily data update completed!\n"
        f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Duration: {hours}h {minutes}m {seconds}s\n"
        f"Updated: {len(updated_tickers)} tickers\n"
        f"Skipped (already up to date): {len(skipped_tickers)} tickers\n"
        f"Errors: {len(error_tickers)} tickers"
    )
    
    console_completion_message = (
        f"Stock/ETF daily data update completed!\n"
        f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Duration: {hours}h {minutes}m {seconds}s\n"
        f"Updated: {len(updated_tickers)} tickers\n"
        f"Skipped (already up to date): {len(skipped_tickers)} tickers\n"
        f"Errors: {len(error_tickers)} tickers"
    )
    
    if error_tickers:
        error_list = f"\nTickers with errors: {', '.join(error_tickers)}"
        telegram_completion_message += error_list
        console_completion_message += error_list
    
    print(console_completion_message)
    send_telegram_message(telegram_completion_message)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Create separate messages for console (no emoji) and Telegram (with emoji)
        # Store emoji message in a variable but don't print it directly
        telegram_error_message = f"ERROR: Unexpected error in Stock/ETF data update: {str(e)}"
        console_error_message = f"ERROR: Unexpected error in Stock/ETF data update: {str(e)}"
        
        print(console_error_message)
        
        # Send error notification to Telegram
        send_telegram_message(telegram_error_message)
        
        # Also log to a file in case this is running from a scheduled task
        log_dir = os.path.join(SCRIPT_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        error_log_path = os.path.join(log_dir, f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"{console_error_message}\n")
            import traceback
            traceback_text = traceback.format_exc()
            f.write(traceback_text)
            
            # Send detailed error to Telegram if not too long
            if len(traceback_text) < 3000:  # Telegram has message length limits
                send_telegram_message(f"Error details:\n{traceback_text}")
        
        print(f"Error details have been logged to: {error_log_path}")
