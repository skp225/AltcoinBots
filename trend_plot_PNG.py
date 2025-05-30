
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import requests
import numpy as np
import logging
import traceback
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info(f"Telegram message sent: {message}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Telegram message: {e}")

try:
    logger.info("Script started")

    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Current directory: {current_dir}")

    # Define paths
    data_dir = os.path.join(current_dir, "Data")
    csv_dir = os.path.join(data_dir, "CSV")
    graphs_dir = os.path.join(current_dir, "Graphs")

    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    logger.info("Directories created/checked")

    # Define the path to the merged CSV file
    file_path = os.path.join(csv_dir, 'dailymergedV3.csv')
    logger.info(f"CSV file path: {file_path}")

    logger.info("Reading the CSV file...")
    data = pd.read_csv(file_path)
    logger.info(f"CSV file read. Shape: {data.shape}")

    # Convert 'DataUpdateDate' to datetime
    data['DataUpdateDate'] = pd.to_datetime(data['DataUpdateDate'], format='%Y-%m-%d')
    logger.info("DataUpdateDate converted to datetime")

    # Filter data to start from June 20th, 2024
    start_date = pd.to_datetime('2024-06-20')
    filtered_data = data[data['DataUpdateDate'] >= start_date]
    logger.info(f"Data filtered. New shape: {filtered_data.shape}")

    # Handle duplicates by taking the mean market cap for each date-project combination
    filtered_data = filtered_data.groupby(['DataUpdateDate', 'name'], as_index=False).agg({'market_cap': 'mean'})
    logger.info("Duplicates handled")

    # Pivot the data to have each project as a column and dates as rows
    pivoted_data = filtered_data.pivot(index='DataUpdateDate', columns='name', values='market_cap')
    logger.info(f"Data pivoted. Shape: {pivoted_data.shape}")

    # Fill missing values (if any) with the previous value (forward fill)
    pivoted_data.fillna(method='ffill', inplace=True)
    logger.info("Missing values filled")

    # Calculate daily percentage change
    pct_change_data = pivoted_data.pct_change().fillna(0)
    logger.info("Daily percentage change calculated")

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
    logger.info("Upward trends identified")

    # Get current date for file naming
    current_date = datetime.now().strftime('%Y%m%d')

    # Updated function to plot trends and return the number of files created
    def plot_trends(trending_projects, period):
        files_created = 0
        projects_per_image = 30
        colors = plt.cm.tab20.colors * 2

        for i in range(0, len(trending_projects), projects_per_image):
            batch = trending_projects[i:i+projects_per_image]
            
            fig, axs = plt.subplots(5, 6, figsize=(32, 28))  # Increased figure size
            
            # Add space at the top for the title
            fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
            
            # Add a bold title above the first row
            fig.suptitle(f'Upward Trending Projects Over {period} Days (Part {files_created + 1})', 
                         fontsize=24, fontweight='bold', y=0.98)
            
            for j, project in enumerate(batch):
                row = j // 6
                col = j % 6
                ax = axs[row, col]
                
                project_data = pivoted_data[project].dropna()
                
                ax.plot(project_data.index, project_data.values, linewidth=2, color=colors[j % len(colors)])
                
                ax.set_title(project, fontsize=10, pad=8)  # Increased padding for project titles
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.tick_params(axis='both', which='minor', labelsize=6)
                
                # Rotate x-axis labels
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')
                
                # Format y-axis labels dynamically
                max_value = project_data.max()
                if max_value >= 1e9:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B'))
                elif max_value >= 1e6:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
                else:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
                
                # Set y-axis limits dynamically
                ax.set_ylim(0, max_value * 1.1)  # 10% headroom
            
            # Remove any unused subplots
            for j in range(len(batch), 30):
                row = j // 6
                col = j % 6
                fig.delaxes(axs[row, col])
            
            plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # Adjust layout while preserving margins
            
            png_file = os.path.join(graphs_dir, f'upward_trends_{period}_days_{current_date}_part{files_created + 1}.png')
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Facet graph for {period}-day upward trends (Part {files_created + 1}) saved as '{png_file}'")
            files_created += 1
        
        return files_created

    # Plot trends for each period
    total_files = 0
    for period in trend_periods:
        logger.info(f"Plotting trends for {period} days")
        files_for_period = plot_trends(trending_projects_dict[period], period)
        total_files += files_for_period
        logger.info(f"Files created for {period} days: {files_for_period}")

    logger.info("All trend plots have been generated and saved in the Graphs folder.")

    # Send Telegram message with information about multiple files
    message = f"Charts available! Total PNG files generated: {total_files}. Check the Graphs folder for multiple parts for each trend period."
    send_telegram_message(message)

    logger.info("Script completed successfully")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    logger.error(traceback.format_exc())
    send_telegram_message(f"An error occurred in trend_plot_PNG.py: {str(e)}")
    sys.exit(1)
