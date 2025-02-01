"""
Data ingestion process.
author: Rui Lu 
date: 12 Jan 2025
"""
#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from datetime import datetime

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get input and output paths
# Load config.json and get input and output paths
with open('src/config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

# Function for data ingestion


def merge_multiple_dataframe():
    # 1: check for datasets, compile them together, and write to an output file
    logging.info('starting data ingestion process')
    # check for datasets
    filenames = next(os.walk(input_folder_path), (None, None, []))[
        2]  # [] if no file
    filenames = [filename for filename in filenames if '\r' not in filename]
    # 2:  read data set
    logging.info(f"Reading files from {input_folder_path}")
    data_list = []
    for file in filenames:
        data_list.append(pd.read_csv(os.path.join(input_folder_path, file)))
    data = pd.concat(data_list)
    # 3. drop duplications
    logging.info("Dropping duplicates")
    data = data.drop_duplicates().reset_index(drop=1)
    # 4. Saving all data set together
    logging.info("Saving ingested meta data")
    data_path = os.path.join(output_folder_path, 'finaldata.csv')
    try:
        data.to_csv(data_path, index=False)
    except FileNotFoundError:
        os.mkdir(output_folder_path)
        data.to_csv(data_path, index=False)
    # 5. Record data ingetion information
    logging.info("Saving ingested meta data")
    record_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(record_path, 'w') as f:
        for file in filenames:
            f.write(file + '\n')
            f.write(
                f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    return (data)


if __name__ == '__main__':
    logging.info("Running ingestion.py")
    merge_multiple_dataframe()
