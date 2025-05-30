import os
import json
import csv
import requests
import pandas as pd
from datetime import datetime

# Configuration
API_URL = "https://api.alternative.me/fng/"
# Use dynamic path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIR, "FearGreedData")
CONSOLIDATED_JSON_FILE = os.path.join(DATA_DIRECTORY, "fear_greed_historical.json")
CONSOLIDATED_CSV_FILE = os.path.join(DATA_DIRECTORY, "fear_greed_historical.csv")
TODAYS_DATE = datetime.today().strftime("%Y-%m-%d")

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
        print(f"Telegram message sent: {message}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Telegram message: {e}")

def fetch_fng_data():
    """Fetch current Fear & Greed Index data from API"""
    try:
        response = requests.get(API_URL, params={"limit": 1})
        response.raise_for_status()
        data = response.json()
        if not data or "data" not in data:
            return None
        return data["data"][0]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FNG data: {str(e)}")
        return None

def load_existing_data():
    """Load existing data from consolidated files if they exist"""
    existing_data = {
        "json": {},
        "csv_rows": []
    }
    
    # Check if JSON file exists and load it
    if os.path.exists(CONSOLIDATED_JSON_FILE):
        try:
            with open(CONSOLIDATED_JSON_FILE, "r") as f:
                json_data = json.load(f)
                if "data" in json_data:
                    existing_data["json"] = json_data
                    print(f"Loaded existing JSON data with {len(json_data['data'])} entries")
        except Exception as e:
            print(f"Error loading existing JSON data: {str(e)}")
    
    # Check if CSV file exists and load it
    if os.path.exists(CONSOLIDATED_CSV_FILE):
        try:
            df = pd.read_csv(CONSOLIDATED_CSV_FILE)
            existing_data["csv_rows"] = df.to_dict('records')
            print(f"Loaded existing CSV data with {len(existing_data['csv_rows'])} entries")
        except Exception as e:
            print(f"Error loading existing CSV data: {str(e)}")
    
    return existing_data

def update_consolidated_data(data):
    """Update consolidated data files with new data"""
    # Create directory if it doesn't exist
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    
    # Load existing data if available
    existing_data = load_existing_data()
    
    # Initialize new data structures
    json_data = existing_data["json"]
    csv_rows = existing_data["csv_rows"]
    
    # If no existing JSON data, create the structure
    if not json_data:
        json_data = {
            "metadata": {
                "source": "Alternative.me Fear & Greed Index",
                "url": "https://alternative.me/crypto/fear-and-greed-index/",
                "last_updated": datetime.now().isoformat()
            },
            "data": {}
        }
    
    # Check if today's data already exists
    if TODAYS_DATE in json_data.get("data", {}):
        print(f"Data for {TODAYS_DATE} already exists in consolidated files, updating...")
    
    # Add date field to entry for CSV
    entry_with_date = data.copy()
    entry_with_date["date"] = TODAYS_DATE
    
    # Update JSON data
    json_data["data"][TODAYS_DATE] = data
    json_data["metadata"]["last_updated"] = datetime.now().isoformat()
    
    # Update CSV rows - remove existing entry for today if it exists
    csv_rows = [row for row in csv_rows if row.get("date") != TODAYS_DATE]
    csv_rows.append(entry_with_date)
    
    # Save consolidated JSON file
    with open(CONSOLIDATED_JSON_FILE, "w") as f:
        json.dump(json_data, f, indent=2)
    
    # Save consolidated CSV file
    if csv_rows:
        # Convert to DataFrame and sort by date
        df = pd.DataFrame(csv_rows)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
        
        # Save to CSV
        df.to_csv(CONSOLIDATED_CSV_FILE, index=False)
    
    print(f"Updated consolidated files with data for {TODAYS_DATE}")

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        script_name = os.path.basename(__file__)
        
        # Send initial notification with emoji for Telegram but plain text for console
        telegram_start_message = f"🔍 Starting Fear & Greed Index data update at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        console_start_message = f"Starting Fear & Greed Index data update at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(console_start_message)
        send_telegram_message(telegram_start_message)
        
        print("Fetching and saving Fear & Greed Index data...")
        data = fetch_fng_data()
        
        if data:
            # Extract the current value and classification
            value = data.get('value', 'N/A')
            classification = data.get('value_classification', 'N/A')
            
            update_consolidated_data(data)
            
            # Calculate execution time
            end_time = datetime.now()
            duration = end_time - start_time
            seconds = duration.total_seconds()
            
            # Create separate messages for console (no emoji) and Telegram (with emoji)
            telegram_completion_message = (
                f"✅ Fear & Greed Index updated successfully!\n"
                f"Date: {TODAYS_DATE}\n"
                f"Current value: {value} ({classification})\n"
                f"Execution time: {seconds:.2f} seconds\n"
                f"Data saved to: {DATA_DIRECTORY}"
            )
            
            console_completion_message = (
                f"Fear & Greed Index updated successfully!\n"
                f"Date: {TODAYS_DATE}\n"
                f"Current value: {value} ({classification})\n"
                f"Execution time: {seconds:.2f} seconds\n"
                f"Data saved to: {DATA_DIRECTORY}"
            )
            
            print(console_completion_message)
            send_telegram_message(telegram_completion_message)
        else:
            telegram_error_message = f"❌ Failed to fetch Fear & Greed data for {TODAYS_DATE}"
            console_error_message = f"Failed to fetch Fear & Greed data for {TODAYS_DATE}"
            print(console_error_message)
            send_telegram_message(telegram_error_message)
    except Exception as e:
        # Create separate messages for console (no emoji) and Telegram (with emoji)
        telegram_error_message = f"❌ ERROR: Unexpected error in Fear & Greed update: {str(e)}"
        console_error_message = f"ERROR: Unexpected error in Fear & Greed update: {str(e)}"
        
        print(console_error_message)
        
        # Send error notification to Telegram
        send_telegram_message(telegram_error_message)
        
        # Log error details
        log_dir = os.path.join(SCRIPT_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        error_log_path = os.path.join(log_dir, f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(error_log_path, 'w') as f:
            f.write(f"{console_error_message}\n")
            import traceback
            traceback_text = traceback.format_exc()
            f.write(traceback_text)
            
            # Send detailed error to Telegram if not too long
            if len(traceback_text) < 3000:  # Telegram has message length limits
                send_telegram_message(f"Error details:\n{traceback_text}")
        
        print(f"Error details have been logged to: {error_log_path}")
