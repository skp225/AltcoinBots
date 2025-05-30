#!/bin/bash

# ECHO COMMAND - tells you what is happening line by line. Contact me for full program!
echo "Starting up!"

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Set the data directory
DATA_DIR="$SCRIPT_DIR/Data"

# Create the Data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Move to data files directory
cd "$DATA_DIR"

echo "Now in Data directory!"

# VARIABLES
# Uppercase by convention
# Letters, numbers, underscores

echo "Getting datas and saving files:"

# Loop through pages from 1 to 31
for PAGE in {1..31}
do
  FILE_NAME="$PAGE.json"
  
  # Check if the file already exists
  if [ -f "$FILE_NAME" ]; then
    echo "P$PAGE already exists, skipping..."
    continue
  fi

  echo "Starting P$PAGE, Getting datas and saving file:"

  # Get data for the current page
  curl -X 'GET' \
    "CONTACT ME "STEVEKPRO@GMAIL.COM"=100=false" \
    -H 'accept: application/json' \
    -o "$FILE_NAME"

  echo "P$PAGE saved :)"
  
  if [ $PAGE -lt 31 ]; then
    echo "Pausing..."

    # Pause the script for a random time between 60 and 120 seconds
    PAUSE_TIME=$((60 + RANDOM % 61))
    sleep $PAUSE_TIME
  fi
done

echo "Operation complete!"
