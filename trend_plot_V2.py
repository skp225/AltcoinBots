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

# Create a DataFrame for plotting
plot_data = pd.melt(pivoted_data.reset_index(), id_vars=['DataUpdateDate'], var_name='Project', value_name='Market Cap')

# Get current date for file naming
current_date = datetime.now().strftime('%Y%m%d')

# Updated function to plot trends and return the number of files created
def plot_trends(trending_projects, period):
    filtered_plot_data = plot_data[plot_data['Project'].isin(trending_projects)]
    
    # Calculate dynamic y-axis ranges
    y_axis_ranges = filtered_plot_data.groupby('Project')['Market Cap'].agg(['min', 'max'])
    
    charts = []
    for project in trending_projects:
        project_data = filtered_plot_data[filtered_plot_data['Project'] == project]
        y_min = y_axis_ranges.loc[project, 'min']
        y_max = y_axis_ranges.loc[project, 'max']
        
        chart = alt.Chart(project_data).mark_line().encode(
            x='DataUpdateDate:T',
            y=alt.Y('Market Cap:Q', scale=alt.Scale(domain=[y_min, y_max]), axis=alt.Axis(format='$,d')),
            color=alt.Color('Project:N', legend=None),
            tooltip=['Project:N', 'DataUpdateDate:T', 'Market Cap:Q']
        ).properties(
            title=project,
            width=150,
            height=100
        )
        charts.append(chart)
    
    # Split charts into groups of 60 (or adjust as needed)
    chart_groups = [charts[i:i+60] for i in range(0, len(charts), 60)]
    
    files_created = 0
    for i, group in enumerate(chart_groups):
        combined_chart = alt.vconcat(*group).resolve_scale(
            y='independent'
        ).properties(
            title=f'Upward Trending Projects Over {period} Days (Part {i+1})'
        )
        
        html_file = os.path.join(graphs_dir, f'upward_trends_{period}_days_{current_date}_part{i+1}.html')
        combined_chart.save(html_file)
        print(f"Facet graph for {period}-day upward trends (Part {i+1}) saved as '{html_file}'")
        files_created += 1
    
    return files_created

# Plot trends for each period
total_files = 0
for period in trend_periods:
    files_for_period = plot_trends(trending_projects_dict[period], period)
    total_files += files_for_period

print("All trend plots have been generated and saved in the Graphs folder.")

# Send Telegram message with information about multiple files
message = f"Graphs available! Total files generated: {total_files}. Check the Graphs folder for multiple parts for each trend period."
send_telegram_message(message)
