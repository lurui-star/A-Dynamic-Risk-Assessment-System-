"""
Main script call app
author: Rui Lu 
date: 20 Jan 2025
"""
import subprocess
import requests
import time
import os
import json
import logging
import sys

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config file
### upload the dependent files 
with open('src/config.json', 'r') as f:
    config = json.load(f)
prod_deployment_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])
test_data_file = os.path.join(test_data_path, 'testdata.csv')

URL = "http://127.0.0.1:5000"

logging.info(f"iniciate app at {URL}")
subprocess.Popen(["python", "src/app.py"])
time.sleep(2)

logging.info( f"Post request /prediction for {test_data_file}")
response_pred = requests.post(f'{URL}/prediction',json={'filepath': test_data_file }).text

logging.info("Get request /scoring")
response_scor = requests.get(f'{URL}/scoring').text

logging.info("Get request /summarystats")
response_stat = requests.get(f'{URL}/summarystats').text

logging.info("Get request /diagnostics")
response_diag = requests.get(f'{URL}/diagnostics').text

logging.info("Generating report text file")
with open(os.path.join(prod_deployment_path, 'apireturns.txt'), 'w') as file:
    file.write('Ingested Data\n\n')
    file.write('Statistics Summary\n')
    file.write(response_stat)
    file.write('\nDiagnostics Summary\n')
    file.write(response_diag)
    file.write('\n\nTest Data\n\n')
    file.write('Model Predictions\n')
    file.write(response_pred)
    file.write('\nModel Score\n')
    file.write(response_scor)