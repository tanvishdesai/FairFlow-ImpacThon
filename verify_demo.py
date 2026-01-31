
import requests
import json

url = "http://localhost:8000/api/precomputed-demo-cases"


# Switch to Recruitment dataset first
print("Switching to Recruitment dataset...")
resp = requests.post("http://localhost:8000/api/dataset/switch", json={"dataset_id": "recruitment"})
print("Switch response:", resp.json())

try:
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Number of demo cases: {len(data.get('demo_cases', []))}")
    if data.get('demo_cases'):
        print("First case features:")
        print(json.dumps(data['demo_cases'][0].get('features'), indent=2))
        print("First case display name:", data['demo_cases'][0].get('display_name'))
except Exception as e:
    print(f"Error: {e}")
