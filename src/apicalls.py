import os
import json
import sys
import logging
import requests

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config file
with open('src/config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])

# Log file paths for debugging
test_data_file = os.path.join(test_data_path, 'testdata.csv')
logging.info(f"Posting request to /prediction with file path: {test_data_file}")

# POST request to /prediction endpoint
try:
    response_pred = requests.post(f'http://127.0.0.1:6000/prediction', json={'filepath': test_data_file})
    if response_pred.status_code == 200:
        logging.info(f"Prediction Response: {response_pred.text}")
    else:
        logging.error(f"Prediction failed with status code {response_pred.status_code}: {response_pred.text}")
except Exception as e:
    logging.error(f"Error occurred while posting prediction request: {str(e)}")

# GET requests to other endpoints
logging.info("Getting request to /scoring")
try:
    response_scor = requests.get(f'http://127.0.0.1:6000/scoring').text
    logging.info(f"Scoring Response: {response_scor}")
except Exception as e:
    logging.error(f"Error occurred while getting scoring request: {str(e)}")

logging.info("Getting request to /summarystats")
try:
    response_stat = requests.get(f'http://127.0.0.1:7000/summarystats').text
    logging.info(f"Summary Stats Response: {response_stat}")
except Exception as e:
    logging.error(f"Error occurred while getting summary stats request: {str(e)}")

logging.info("Getting request to /diagnostics")
try:
    response_diag = requests.get(f'http://127.0.0.1:6000/diagnostics').text
    logging.info(f"Diagnostics Response: {response_diag}")
except Exception as e:
    logging.error(f"Error occurred while getting diagnostics request: {str(e)}")

# Generate report
logging.info("Generating report text file")
try:
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
    logging.info(f"Report generated at {os.path.join(prod_deployment_path, 'apireturns.txt')}")
except Exception as e:
    logging.error(f"Error occurred while generating the report: {str(e)}")
