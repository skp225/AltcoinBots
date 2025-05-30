import pandas as pd
import os
import datetime as dt
import subprocess
import requests
import json
import time

print("Dependencies loaded!")
time.sleep(2)

# Telegram configuration
TELEGRAM_TOKEN = ''
CHAT_ID = ''

def send_telegram_message(message):
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

# Function to clear temp files (old JSON files)
def clean_up():
    os.chdir("C:/Users/user/Dropbox/Curl Gecko/AltCoinResearch/Data")
    print("Now inside data directory for cleaning temp files....")
    time.sleep(2)

    for i in range(1, 32):
        try:
            os.remove(f"{i}.json")
        except FileNotFoundError:
            pass
    print("Rouge Files From Previous Operations Deleted!")
    time.sleep(2)

# Function to perform all data processing tasks
def all_processes():
    os.chdir("C:/Users/user/Dropbox/Curl Gecko/AltCoinResearch")
    subprocess.call("cgpulldailyV8.sh", shell=True)
    print("Data Pull Complete! Next Operation: Data Processing!")

    os.chdir("C:/Users/user/Dropbox/Curl Gecko/AltCoinResearch/Data")
    print("Now inside that directory!")

    # Load JSON files into dataframes
    dataframes = []
    for i in range(1, 32):
        df = pd.read_json(f"{i}.json")
        df['DataUpdateDate'] = df['last_updated'].str[:10]
        dataframes.append(df)

    print("Files are all loaded...")

    # Create the final dataframe
    print("Starting Next Operation: Merging and removing duplicates")
    df_final = pd.concat(dataframes).sort_values('market_cap').drop_duplicates(subset=['id', 'last_updated'], keep='last')
    print("Data has been merged and is ready for export!")

    # Define today for file output naming purposes
    today = pd.Timestamp('today')

    os.chdir("C:/Users/user/Dropbox/Curl Gecko/AltCoinResearch/Data/CSV")
    print("Now inside data dump directory!")

    df_final.to_csv(f'CGdata_{today:%m%d%Y}.csv', index=False)
    print("File exported!")

    os.chdir("C:/Users/user/Dropbox/Curl Gecko/AltCoinResearch/Data")
    print("Now back inside temp data directory!")

    # Delete temp JSON files
    print("Starting Next Operation: Deleting Temporary Files")
    clean_up()
    print("Files deleted! Operation Complete!")

# Function to run all processes immediately
def run_immediately():
    print("Running all processes immediately...")
    all_processes()
    send_telegram_message('All processes successful')

# Main code execution
print("Starting up all processes...")
time.sleep(1)
run_immediately()
print("All processes successful")
time.sleep(1)
print("Closing now...")
time.sleep(1)
exit()