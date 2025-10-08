#!/usr/bin/env python3

import flask
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = flask.Flask(__name__)
flag = open("/flag").read().strip()
index_html = """
<html>
<head>
    <title>Credit Check</title>
</head>
<body>
    <h1>Credit Check</h1>
    <a href="static/Mall_Customers.csv" target="_blank">Download Data</a>
    <form action="check" method="post" enctype="multipart/form-data">
        <div class="form-group">
            <label for="data_file"><b>Upload Data:</b></label>
            <input type="file" class="form-control-file" name="data_file" required>
        </div>
        <button type="submit" class="btn btn-success" value="Upload">Upload</button>
    </form>
</body>
</html>
"""

class Predictor:
    def __init__(self):
        self.reg = LinearRegression()
        self.scaler = StandardScaler()

    def train(self, df: pd.DataFrame):
        df[["Age", "Annual Income (k$)"]] = self.scaler.fit_transform(df[["Age", "Annual Income (k$)"]].values)
        self.reg.fit(df[["Age", "Annual Income (k$)"]].values, df["Spending Score (1-100)"])

    def predict(self, scaled_age_income: np.ndarray) -> float:
        return self.reg.predict(scaled_age_income)

@app.route("/")
def index():
    return flask.make_response(index_html, 200)

@app.route("/check", methods=["POST"])
def check():
    try:
        with flask.request.files["data_file"].stream as stream:
            tampered_df = pd.read_csv(stream)
    except:
        return flask.make_response("Bad data", 400)
    standard_df = pd.read_csv("/challenge/static/Mall_Customers.csv")
    change = len(pd.concat([standard_df, tampered_df]).drop_duplicates(keep=False))
    if change > 1:
        return flask.make_response("Your tampering was too flagrant. You were discovered.", 200)
    p = Predictor()
    p.train(tampered_df)
    score = p.predict(p.scaler.transform(np.array([[19, 15]])))
    result = "You don't qualify. Try again." if score < 90 else f"Enjoy your Platinum Mall Card: {flag}"
    return flask.make_response(result, 200)
