"""
Train simple model 
author: Rui Lu 
date: 12 Jan 2025
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
import logging
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

# Load config.json and get path variables
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
with open('src/config.json', 'r') as f:
    config = json.load(f)
dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])

# Function for training the model


def train_model():
    # 1: loading data set
    logging.info("Loading finaldata.csv")
    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    y_df = data_df.pop('exited')
    X_df = data_df.drop(['corporation'], axis=1)
    # 2. Fit logistic regression model
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)
    logging.info("Training model")
    model.fit(X_df, y_df)
    # 3 save model
    logging.info("Saving trained model")
    pickle.dump(model, open(os.path.join(
        model_path, 'trainedmodel.pkl'), 'wb'))


if __name__ == '__main__':
    logging.info("Running training.py")
    train_model()
