import re
import subprocess
import pandas as pd
from flask import Flask, jsonify, request
import diagnostics

######################Set up variables for use in our script
# initiate the app
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

#######################Prediction Endpoint
@app.route('/')
def index():
    user = request.args.get('user')
    return "Welcome " + user + '\n'

#######################Scoring Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """
    Prediction endpoint that loads data given the file path
    and calls the prediction function in diagnostics.py

    Returns:
        json: model predictions
    """
    filepath = request.get_json()['filepath']
    
    # Load the data
    df = pd.read_csv(filepath)
    df = df.drop(['corporation', 'exited'], axis=1)

    # Get predictions from diagnostics
    preds = diagnostics.model_predictions(df)

    preds = preds.tolist()  # Convert pandas Series to list

    return jsonify(preds)

@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """
    Scoring endpoint that runs the script scoring.py and
    gets the score of the deployed model

    Returns:
        str: model f1 score
    """
    output = subprocess.run(['python', 'src/scoring.py'], capture_output=True).stdout
    output = re.findall(r'f1 score = \d*\.?\d+', output.decode())[0]
    return output

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """
    Summary statistics endpoint that calls dataframe summary
    function from diagnostics.py

    Returns:
        json: summary statistics
    """
    summary_stats = diagnostics.dataframe_summary()
    
    # Convert DataFrame to JSON serializable format (if needed)
    if isinstance(summary_stats, pd.DataFrame):
        summary_stats = summary_stats.to_dict(orient='records')  # Convert to list of dicts
    
    return jsonify(summary_stats)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diag():
    """
    Diagnostics endpoint that calls missing_percentage, execution_time,
    and outdated_package_list from diagnostics.py

    Returns:
        dict: missing percentage, execution time, and outdated packages
    """
    missing = diagnostics.missing_percentage()
    time = diagnostics.execution_time()
    outdated = diagnostics.outdated_packages_list()

    ret = {
        'missing_percentage': missing,
        'execution_time': time,
        'outdated_packages': outdated
    }

    return jsonify(ret)


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
