#!/bin/bash

BASE_URL="https://mesonet.agron.iastate.edu/pickup/wwa" # URL for vtec files.
OUTPUT_DIR="$(pwd)/data/VTEC"

mkdir -p "$OUTPUT_DIR"

# Only doing 2014 - 2024 as we only have power-outage data from these years.
for YEAR in {2014..2024}; do
    FILE_URL="${BASE_URL}/${YEAR}_all.zip" 
    OUTPUT_FILE="${OUTPUT_DIR}/file_${YEAR}_all.zip"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping: $OUTPUT_FILE (already exists)"
    else    
        echo "Downloading $FILE_URL..."
        wget -q -O "$OUTPUT_FILE" "$FILE_URL" && echo "Downloaded: $OUTPUT_FILE" || echo "Failed: $FILE_URL"
    fi
done

echo "Download complete"
