"""
Score the model 
author: Rui Lu 
date: 12 Jan 2025
"""
import pandas as pd
import pickle
import os
import sys
import logging
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
#################Load config.json and get path variables
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
with open('src/config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 

#################Function for model scoring
def score_model():
    # 1. read test data 
    logging.info("Loading testdata.csv")
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    # 2. load model 
    logging.info("Loading trained model")
    model = pickle.load(open( os.path.join(model_path, 'trainedmodel.pkl'),'rb'))
    # 3. prepare test data  
    logging.info("Preparing test data")
    y_true = test_df.pop('exited')
    X_df = test_df.drop(['corporation'], axis=1)
    # 4. predict test data  
    logging.info("Predicting test data")
    y_pred = model.predict(X_df)
    f1 = f1_score(y_true, y_pred)
    print(f"f1 score = {f1}")
    # 5. record information 
    logging.info("Saving scores to text file")
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as file:
        file.write(f"f1 score = {f1}")

if __name__ == '__main__':
    logging.info("Running scoring.py")
    score_model()