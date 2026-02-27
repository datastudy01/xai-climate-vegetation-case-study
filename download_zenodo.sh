#!/bin/bash

# Hardcoded record ID and filenames
RECORD_ID="18434535"
DOWNLOAD_DIR="./datasets"
FILES=("temperate" "boreal" "tropical")

if [ -z "$ZENODO_TOKEN" ]; then
  echo "ZENODO_TOKEN environment variable not set."
  exit 1
fi

for FILENAME in "${FILES[@]}"; do
  PYTHON_CMD="from utils.download_zenodo_file import download_zenodo_file; download_zenodo_file('$RECORD_ID', '{}.zip'.format('$FILENAME'), extract_to='$DOWNLOAD_DIR')"
  "$(which python)" -c "$PYTHON_CMD"
done
