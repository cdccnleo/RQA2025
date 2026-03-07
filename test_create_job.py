import urllib.request
import urllib.parse
import json

data = {
    "model_type": "RandomForest",
    "config": {
        "symbols": ["002837", "688702", "000987"],
        "epochs": 5
    }
}

req = urllib.request.Request(
    "http://localhost/api/v1/ml/training/jobs",
    data=json.dumps(data).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as response:
        print(response.status)
        print(response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code}")
    print(e.read().decode('utf-8'))
