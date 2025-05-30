import pandas as pd
import os
import glob

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths
data_dir = os.path.join(current_dir, "Data")
csv_dir = os.path.join(data_dir, "CSV")

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# Use the CSV directory path
path = csv_dir

all_files = glob.glob(os.path.join(path, "CGdata*.csv"))

all_df = []
for f in all_files:
    df = pd.read_csv(f, sep=',', encoding='latin-1')
    df['file'] = os.path.basename(f)  # Use os.path.basename instead of split
    all_df.append(df)

merged_df = pd.concat(all_df, ignore_index=True, sort=True)

# Save the merged file in the CSV directory
output_file = os.path.join(csv_dir, 'dailymerged.csv')
merged_df.to_csv(output_file, index=False)

print(f'Dataset merge complete! Saved as {output_file}')
