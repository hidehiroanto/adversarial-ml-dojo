import json
import requests

with open('test_pitch.json') as f:
    test = json.load(f)

r = requests.post('http://localhost/score', json=test)
print(r.json())
