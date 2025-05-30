#!/usr/bin/env python3
"""
Check and Run Lenient Trend Analysis

This script:
1. Checks the most recent 3-day trending Excel file
2. Counts the number of projects in the file
3. If there are fewer than 20 projects, runs trend_analysis_lenient.py to create more comprehensive trending files

Usage:
    python check_and_run_lenient.py [min_projects]

Arguments:
    min_projects: Optional. Minimum number of projects required (default: 20)
"""

import os
import sys
import glob
import subprocess
from datetime import datetime, timedelta
import traceback

# Try to import required packages, install if missing
try:
    import pandas as pd
    import requests
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Attempting to install dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas requests openpyxl"])
        import pandas as pd
        import requests
        print("Dependencies installed successfully.")
    except Exception as install_error:
        print(f"Failed to install dependencies: {install_error}")
        print("Please install required packages manually: pandas, requests, openpyxl")
        sys.exit(1)

# Telegram configuration (same as in other scripts)
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

def find_most_recent_trending_file():
    """Find the most recent 3-day trending Excel file."""
    print("Looking for the most recent 3-day trending file...")
    
    # Path to the Excel output directory
    excel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Excel")
    
    # Check if the Excel directory exists
    if not os.path.exists(excel_dir):
        print(f"Excel directory not found: {excel_dir}")
        return None
    
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
    
    # Look for 3-day trending files from the last few days
    for date_str in dates:
        file_path = os.path.join(excel_dir, f"3days_trending_{date_str}.xlsx")
        if os.path.exists(file_path):
            print(f"Found 3-day trending file from {date_str}: {file_path}")
            return file_path
    
    # If no specific 3-day trending files found, look for any 3-day trending files
    all_3day_files = glob.glob(os.path.join(excel_dir, "3days_trending_*.xlsx"))
    
    if all_3day_files:
        # Sort by modification time (newest first)
        all_3day_files.sort(key=os.path.getmtime, reverse=True)
        newest_file = all_3day_files[0]
        print(f"Found 3-day trending file: {newest_file}")
        return newest_file
    
    print("No 3-day trending files found.")
    return None

def count_projects_in_file(file_path):
    """Count the number of projects in an Excel file."""
    try:
        df = pd.read_excel(file_path)
        project_count = len(df)
        print(f"Found {project_count} projects in {os.path.basename(file_path)}")
        return project_count
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return 0

def run_lenient_analysis():
    """Run the lenient trend analysis script."""
    print("Running lenient trend analysis...")
    
    # Get the path to the trend_analysis_lenient.py script
    lenient_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trend_analysis_lenient.py")
    
    # Check if the script exists
    if not os.path.exists(lenient_script_path):
        error_msg = f"Error: Lenient trend analysis script not found at {lenient_script_path}"
        print(error_msg)
        send_telegram_message(f"❌ {error_msg}")
        return False
    
    # Run the script as a subprocess
    try:
        subprocess.run([sys.executable, lenient_script_path], check=True)
        print("Lenient trend analysis completed successfully.")
        send_telegram_message("✅ Lenient trend analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"Error running lenient trend analysis: {e}"
        print(error_msg)
        send_telegram_message(f"❌ {error_msg}")
        return False

def rename_lenient_files():
    """Rename lenient trending files to replace the regular trending files."""
    print("Renaming lenient trending files to replace regular trending files...")
    
    # Path to the Excel output directory
    excel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Excel")
    
    # Get the current date
    current_date = datetime.now().strftime('%Y%m%d')
    
    # Find all lenient trending files
    lenient_files = glob.glob(os.path.join(excel_dir, "*days_trending_lenient_*.xlsx"))
    
    if not lenient_files:
        print("No lenient trending files found.")
        return False
    
    # Track renamed files
    renamed_files = []
    
    # Process each lenient file
    for lenient_file in lenient_files:
        try:
            # Extract the period (e.g., 3days) from the filename
            file_name = os.path.basename(lenient_file)
            parts = file_name.split('_')
            
            if len(parts) >= 4 and parts[0].endswith('days') and parts[1] == 'trending' and parts[2] == 'lenient':
                period = parts[0]  # e.g., 3days
                
                # Create the new filename (without "lenient")
                new_file_name = f"{period}_trending_{current_date}.xlsx"
                new_file_path = os.path.join(excel_dir, new_file_name)
                
                # Rename the file (effectively overwriting the original file if it exists)
                os.replace(lenient_file, new_file_path)
                print(f"Renamed {file_name} to {new_file_name}")
                renamed_files.append(new_file_name)
        except Exception as e:
            print(f"Error renaming {lenient_file}: {e}")
    
    if renamed_files:
        message = f"✅ Replaced {len(renamed_files)} trending files with lenient versions"
        print(message)
        send_telegram_message(message)
        return True
    else:
        message = "❌ No files were renamed"
        print(message)
        send_telegram_message(message)
        return False

def main():
    """Main function."""
    start_time = datetime.now()
    script_name = os.path.basename(__file__)
    
    # Get minimum projects threshold from command line argument or use default
    min_projects = 20
    if len(sys.argv) > 1:
        try:
            min_projects = int(sys.argv[1])
        except ValueError:
            print(f"Invalid argument: {sys.argv[1]}. Using default minimum of {min_projects} projects.")
    
    # Send initial Telegram notification
    start_message = f"🔍 Starting {script_name} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    print(start_message)
    send_telegram_message(start_message)
    
    try:
        # Find the most recent 3-day trending file
        trending_file = find_most_recent_trending_file()
        
        if trending_file is None:
            message = "❌ No trending files found. Running lenient analysis..."
            print(message)
            send_telegram_message(message)
            run_lenient_analysis()
            rename_lenient_files()
        else:
            # Count the number of projects in the file
            project_count = count_projects_in_file(trending_file)
            
            # Check if we need to run the lenient analysis
            if project_count < min_projects:
                message = f"⚠️ Only {project_count} projects found in {os.path.basename(trending_file)}. Minimum required: {min_projects}. Running lenient analysis..."
                print(message)
                send_telegram_message(message)
                
                # Run the lenient analysis
                if run_lenient_analysis():
                    # Rename the lenient files to replace the regular files
                    rename_lenient_files()
            else:
                message = f"✅ Found {project_count} projects in {os.path.basename(trending_file)}. Minimum required: {min_projects}. No need to run lenient analysis."
                print(message)
                send_telegram_message(message)
        
        # Calculate execution time
        end_time = datetime.now()
        duration = end_time - start_time
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Send completion notification
        completion_message = (
            f"✅ {script_name} completed successfully!\n"
            f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Duration: {hours}h {minutes}m {seconds}s"
        )
        print("\n" + completion_message)
        send_telegram_message(completion_message)
        
    except Exception as e:
        # Send error notification
        error_message = f"❌ Error in {script_name}: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        send_telegram_message(error_message)
        sys.exit(1)

if __name__ == "__main__":
    main()
