"""
Deploythe model 
author: Rui Lu 
date: 12 Jan 2025
"""
import pandas as pd
import pickle
import os
import logging
import sys
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil

##################Load config.json and correct path variable
with open('src/config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
####################function for deployment
def deploy_model():
    logging.info("Deploying trained model to production")
    logging.info("Copying trainedmodel.pkl, ingestfiles.txt and latestscore.txt")
    shutil.copy(os.path.join(dataset_csv_path,'ingestedfiles.txt'),prod_deployment_path)
    shutil.copy(os.path.join(model_path,'trainedmodel.pkl'),prod_deployment_path)
    shutil.copy(os.path.join(model_path,'latestscore.txt'),prod_deployment_path)
        
if __name__ == '__main__':
    logging.info("Running deployment.py")
    deploy_model()

