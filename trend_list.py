#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import altair as alt
import os
from datetime import datetime
import requests

# Telegram configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

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

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths
data_dir = os.path.join(current_dir, "Data")
csv_dir = os.path.join(data_dir, "CSV")
graphs_dir = os.path.join(current_dir, "Graphs")

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

# Define the path to the merged CSV file
file_path = os.path.join(csv_dir, 'dailymergedV3.csv')

print("Reading the CSV file...")
data = pd.read_csv(file_path)

# Convert 'DataUpdateDate' to datetime
data['DataUpdateDate'] = pd.to_datetime(data['DataUpdateDate'], format='%Y-%m-%d')

# Filter data to start from June 20th, 2024
start_date = pd.to_datetime('2024-06-20')
filtered_data = data[data['DataUpdateDate'] >= start_date]

# Handle duplicates by taking the mean market cap for each date-project combination
filtered_data = filtered_data.groupby(['DataUpdateDate', 'name'], as_index=False).agg({'market_cap': 'mean'})

# Pivot the data to have each project as a column and dates as rows
pivoted_data = filtered_data.pivot(index='DataUpdateDate', columns='name', values='market_cap')

# Fill missing values (if any) with the previous value (forward fill)
pivoted_data.fillna(method='ffill', inplace=True)

# Calculate daily percentage change
pct_change_data = pivoted_data.pct_change().fillna(0)

# Function to identify projects with upward trends over a specified period
def identify_upward_trends(data, period):
    trending_projects = []
    for project in data.columns:
        ts = data[project].dropna()
        # Check if the latest date has been trending upward for the given period
        positive_days = sum(ts[-period:] > 0)
        if positive_days >= period:
            trending_projects.append(project)
    return trending_projects

# Identify projects with upward trends for different periods
trend_periods = [2, 3, 5, 7, 10, 14, 21, 30]
trending_projects_dict = {period: identify_upward_trends(pct_change_data, period) for period in trend_periods}

# Create a DataFrame for reference with list of project names identified, and trend period
main_df = pd.DataFrame({f'{period}_day_trend': trending_projects_dict[period] for period in trend_periods})
main_df['TrendPeriod'] = main_df.index + 2

# Get current date for file naming
current_date = datetime.now().strftime('%Y%m%d')

# Export the DataFrame to CSV files in the Graphs folder with the current date
for period, trending_projects in trending_projects_dict.items():
    trend_file_path = os.path.join(graphs_dir, f'trending_{period}_days_{current_date}.csv')
    main_df[['TrendPeriod', str(period) + '_day_trend']].to_csv(trend_file_path, index=False)


# Send Telegram message
send_telegram_message("Lists available!")
