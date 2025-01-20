"""
Diagnosis of the model 
author: Rui Lu 
date: 12 Jan 2025
"""
import os
import sys
import json
import pickle
import timeit
import logging
import subprocess
import numpy as np
import pandas as pd
import re
import yaml

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('src/config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path']) 
data_path=os.path.join(config['output_folder_path']) 
test_data_path=os.path.join(config['test_data_path']) 

def model_predictions(X_df):
    logging.info("Loading deployed model")
    model = pickle.load(open(os.path.join(prod_deployment_path,'trainedmodel.pkl'), 'rb'))
    logging.info("Running predictions on data")
    y_pred = model.predict(X_df)
    return y_pred

def dataframe_summary():
    logging.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(data_path, 'finaldata.csv'))
    data_df = data_df.drop(['exited'], axis=1)
    data_df = data_df.select_dtypes('number')
    logging.info("Calculating statistics for data")
    statistics_dict = {}
    for col in data_df.columns:
        mean = data_df[col].mean()
        median = data_df[col].median()
        std = data_df[col].std()
        statistics_dict[col] = {'mean': mean, 'median': median, 'std': std}
    return(statistics_dict)


def missing_percentage():
    logging.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(data_path, 'finaldata.csv'))
    logging.info("Calculating missing data percentage")
    missing_list = {col: {'percentage': perc} for col, perc in zip(
        data_df.columns, data_df.isna().sum() / data_df.shape[0] * 100)}

    return( missing_list)


def _ingestion_timing():
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    timing = timeit.default_timer() - starttime
    return timing

def _training_timing():
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'training.py'], capture_output=True)
    timing = timeit.default_timer() - starttime
    return timing

def execution_time():
    logging.info("Calculating time for ingestion.py")
    ingestion_time = []
    for _ in range(20):
        time = _ingestion_timing()
        ingestion_time.append(time)

    logging.info("Calculating time for training.py")
    training_time = []
    for _ in range(20):
        time = _training_timing()
        training_time.append(time)

    ret_list = [
        {'ingest_time_mean': np.mean(ingestion_time)},
        {'train_time_mean': np.mean(training_time)}
    ]

    return ret_list

def outdated_packages_list(request_file='requirements.txt'):
    logging.info("Checking outdated dependencies")
    
    # Step 1: Run pip list to get outdated packages
    result = subprocess.run(
        ['pip', 'list', '--outdated'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8'
    )

    # Check if the subprocess ran successfully
    if result.returncode != 0:
        logging.error(f"Error running pip list: {result.stderr}")
        return []

    dep = result.stdout
    # Split lines and ignore the first 2 (header lines)
    dep_lines = dep.split('\n')[2:]  # Skip the first two lines of the output header

    # Step 2: Parse the pip list output
    outdated_deps = []
    for line in dep_lines:
        if line.strip():  # Skip empty lines
            parts = line.split()
            if len(parts) >= 3:
                # Package name, current version, latest version
                outdated_deps.append({
                    'package': parts[0],
                    'current_version': parts[1],
                    'latest_version': parts[2]
                })
    
    # Convert the outdated list to a DataFrame
    outdated_deps_df = pd.DataFrame(outdated_deps)
    
    # Step 3: Read the request.txt file to get the list of packages
    try:
        with open(request_file, 'r') as f:
            requested_packages = [line.strip().split('==')[0] for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"File {request_file} not found.")
        return outdated_deps_df

    # Step 4: Filter outdated packages that are in request.txt
    requested_outdated_deps = outdated_deps_df[outdated_deps_df['package'].isin(requested_packages)]
    
    # Return only the outdated packages that are in request.txt, without extra print/logging
    return requested_outdated_deps

if __name__ == '__main__':

    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X_df = test_df.drop(['corporation', 'exited'], axis=1)

    print("Model predictions on testdata.csv:",
          model_predictions(X_df), end='\n\n')

    print("Summary statistics")
    print(json.dumps(dataframe_summary(), indent=4), end='\n\n')

    print("Missing percentage")
    print(json.dumps(missing_percentage(), indent=4), end='\n\n')

    print("Execution time")
    print(json.dumps(execution_time(), indent=4), end='\n\n')

    print("Outdated Packages")
    print(outdated_packages_list(request_file='requirements.txt'))