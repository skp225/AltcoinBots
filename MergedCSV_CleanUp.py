#!/usr/bin/env python
# coding: utf-8

# Import dependencies
import pandas as pd
import os
import csv
import datetime as dt
import schedule
import time
import subprocess
import json

print("Dependencies loaded!")

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths
data_dir = os.path.join(current_dir, "Data")
csv_dir = os.path.join(data_dir, "CSV")

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# Move to the CSV directory
os.chdir(csv_dir)
print("Now inside data csv directory!")

print("Starting Next Operation: Import Data")

# Define filepath
filepath = os.path.join(csv_dir, "dailymerged.csv")

# Read the CSV file
df = pd.read_csv(filepath, low_memory=False)
print("Merged file loaded!")

print("Removing duplicates!")

df2 = df.drop_duplicates(['DataUpdateDate', 'market_cap', 'id'], keep='first')

print("Removing File column!")
if 'file' in df2.columns:
    del df2['file']

print("Removing general duplicates (extra headers)!")

df3 = df2.drop_duplicates(subset=None, keep=False, inplace=False)

# Save the cleaned data
output_file = os.path.join(csv_dir, 'dailymergedV2.csv')
df3.to_csv(output_file, index=False)

print(f"Clean Datafile available at: {output_file}")

print("Deleting Original Merge File.")

# Try to delete the original merge file
try:
    os.remove(filepath)
    print(f"The file: {filepath} is deleted!")
except OSError as e:
    print(f"Error: {e.filename} - {e.strerror}!")

print("Clean-up process completed!")
