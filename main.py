"""
Main script host different steps 
author: Rui Lu 
date: 20 Jan 2025
"""
import os
import re
import sys
import logging
import pandas as pd
from sklearn.metrics import f1_score
import json

from src import scoring
from src import training
from src import ingestion
from src import reporting
from src import deployment
from src import diagnostics

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


### Config information 
with open('src/config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
data_path=os.path.join(config['output_folder_path']) 


def main():
    # Step I: Check for new data
    logging.info("Checking for new data...")
    ingested_files = set(open(os.path.join(prod_deployment_path, "ingestedfiles.txt")).readlines()[1:])
    source_files = set(os.listdir(input_folder_path))
    
    # If no new data, exit
    if not source_files.difference(ingested_files):
        logging.info("No fresh data found.")
        return

    # Step II: Ingest new data
    logging.info("Ingesting new data...")
    ingestion.merge_multiple_dataframe()

    # Step III: Check for model drift
    logging.info("Checking for model drift...")
    deployed_score = float(re.findall(r'\d*\.?\d+', open(os.path.join(prod_deployment_path, "latestscore.txt")).read())[0])
    data_df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    X_df, y_df = data_df.drop(['corporation', 'exited'], axis=1), data_df['exited']
    y_pred = diagnostics.model_predictions(X_df)
    new_score = f1_score(y_df, y_pred)

    logging.info(f"Deployed score: {deployed_score}, New score: {new_score}")

    if new_score >= deployed_score:
        logging.info("No model drift detected.")
        return

    # Step IV: Re-train and re-deploy model
    logging.info("Model drift detected. Re-training and re-deploying model...")
    training.train_model()
    scoring.score_model()
    deployment.deploy_model()

    # Step V: Run diagnostics and reporting
    logging.info("Generating diagnostics and reports...")
    reporting.plot_confusion_matrix()
    reporting.generate_pdf_report(prod_deployment_path, data_path)

if __name__ == '__main__':
    main()