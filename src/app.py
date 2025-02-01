"""
Main script host app
author: Rui Lu 
date: 20 Jan 2025
"""

import os
import re
import subprocess
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import diagnostics

# Initialize Flask app
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'
CORS(app)  # Enable CORS for all routes

# Base URL
URL = 'http://127.0.0.1:5000'

# Prediction Endpoint


@app.route('/')
def index():
    # Default to 'Guest' if no user is provided
    user = request.args.get('user', 'Guest')
    return f"Welcome {user}\n"

# Prediction Endpoint


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """
    Prediction endpoint that loads data from the given file path
    and calls the model prediction function in diagnostics.py.

    Returns:
        json: model predictions
    """
    # Extract file path from incoming request
    filepath = request.get_json().get('filepath')
    # Read CSV and drop unwanted columns
    df = pd.read_csv(filepath)
    # Ignore errors if columns not found
    df = df.drop(['corporation', 'exited'], axis=1, errors='ignore')
   # Call diagnostics to get predictions
    preds = diagnostics.model_predictions(df)
    return (jsonify(preds.tolist()))

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """
    Scoring endpoint that runs the scoring script and
    returns the model's F1 score.

    Returns:
        str: model f1 score
    """
    output = subprocess.run(['python', 'src/scoring.py'],
                            capture_output=True, text=True)
    # Extract the f1 score from the output
    f1_score_match = re.findall(r'f1 score = \d*\.?\d+', output.stdout)

    return jsonify({"f1_score": f1_score_match[0]})

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """
    Summary statistics endpoint that calls the dataframe summary
    function from diagnostics.py.

    Returns:
        json: summary statistics
    """
    # Get summary statistics from diagnostics
    summary_stats = diagnostics.dataframe_summary()

    # Check if summary_stats is a dictionary (it should be)
    if isinstance(summary_stats, dict):
        # No need to convert to dict since it's already a dictionary
        pass
    else:
        # Handle unexpected type (raise an error or return an empty response)
        return jsonify({"error": "Unexpected data type from dataframe_summary"}), 500

    # Return the summary statistics as JSON
    return jsonify(summary_stats)

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diag():
    """
    Diagnostics endpoint that returns missing data percentage,
    execution time, and outdated packages from diagnostics.py.

    Returns:
        dict: missing percentage, execution time, and outdated packages
    """

    # Get missing data percentage, execution time, and outdated packages from diagnostics
    missing = diagnostics.missing_percentage()
    time = diagnostics.execution_time()
    outdated = diagnostics.outdated_packages_list()

    # Convert execution time into a serializable format (list of dictionaries)
    time_serializable = []
    for item in time:
        # Since each item has either 'ingest_time_mean' or 'train_time_mean',
        # we can directly append it to the list.
        for key in item:
            time_serializable.append({key: float(item[key])})

    # Convert the outdated packages DataFrame into a serializable format (list of dicts)
    outdated_serializable = outdated.to_dict(orient='records')

    # Compile the response
    ret = {
        'missing_percentage': missing,
        'execution_time': time_serializable,
        'outdated_packages': outdated_serializable
    }

    # Return the response as JSON
    return jsonify(ret)


# Run the Flask App
if __name__ == "__main__":
    # Run the Flask app
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
