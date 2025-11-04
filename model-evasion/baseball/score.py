#!/usr/bin/exec-suid -- /usr/bin/python3 -I

import flask
import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import warnings

warnings.filterwarnings('ignore')
app = flask.Flask(__name__)

def get_classification(pitches):
    gpc = joblib.load('/challenge/baseball_gpc.joblib')
    predicted_user = gpc.predict(pitches.flatten().reshape(1, -1))
    proba = gpc.predict_proba(pitches.flatten().reshape(1, -1)).max()
    if predicted_user == 'henry':
        if proba > 0.7:
            flag = open('/flag').read()
            return f'Bring the heat! {flag}'
        else:
            return f"You look like henry, but we're not confident enough. Confidence: {proba}."
    else:
        return f'You throw like {predicted_user} with a likelihood of {proba}'

def get_update(pitches):
    pitch_map = np.zeros((30, 30))
    for pitch in pitches:
        pitch_map[pitch[0], pitch[1]] = 1
    return pitch_map

@app.route('/score', methods=['POST'])
def run():
    try:
        input_data = flask.request.get_json()
        pitch_list = list(set([(int(i['x']), int(i['y'])) for i in input_data['pitches']]))
    except Exception:
        flask.abort(400, 'Bad data')
    if len(pitch_list) != 15:
        message = 'Not 15 unique pitches. Try again.'
    elif any(min(x) < 0 or max(x) > 29 for x in pitch_list):
        message = 'Pitch out of bounds. Keep them between 0 and 29 (inclusive).'
    else:
        message = get_classification(get_update(pitch_list))
    return flask.jsonify({'message': message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
