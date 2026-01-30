
import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_api():
    print("🔍 Testing Dataset Switching API...")
    
    # 1. Get initial datasets
    print("\n1. Fetching datasets...")
    res = requests.get(f"{BASE_URL}/api/datasets")
    assert res.status_code == 200
    datasets = res.json()
    print(f"   Datasets: {[d['id'] for d in datasets]}")
    active = next(d for d in datasets if d['active'])
    print(f"   Active: {active['id']}")
    assert active['id'] == 'adult'

    # 2. Switch to German Credit
    print("\n2. Switching to German Credit...")
    res = requests.post(f"{BASE_URL}/api/dataset/switch", json={"dataset_id": "german"})
    assert res.status_code == 200
    print("   Switch successful.")
    
    # 3. Verify active dataset
    res = requests.get(f"{BASE_URL}/api/datasets")
    active = next(d for d in res.json() if d['active'])
    print(f"   New Active: {active['id']}")
    assert active['id'] == 'german'
    
    # 4. Check available models
    print("\n3. Fetching models for German Credit...")
    res = requests.get(f"{BASE_URL}/api/models")
    models = res.json()
    print(f"   Models: {[m['id'] for m in models]}")
    assert len(models) >= 3
    
    # 5. Make a prediction (dummy data)
    print("\n4. Making a prediction...")
    # German credit features (approximate)
    dummy_features = {
        "Age": 30, "Sex": "male", "Job": 2, "Housing": "own", 
        "Saving accounts": "little", "Checking account": "moderate", 
        "Credit amount": 1000, "Duration": 12, "Purpose": "radio/TV"
    }
    res = requests.post(f"{BASE_URL}/api/predict", json={"features": dummy_features})
    if res.status_code == 200:
        pred = res.json()
        print(f"   Prediction: {pred['base_prediction']} -> {pred['fairflow_decision']}")
        print(f"   Intervention: {pred['intervention_type']}")
    else:
        print(f"❌ Prediction failed: {res.text}")
        
    print("\n✅ Verification Complete!")

if __name__ == "__main__":
    try:
        test_api()
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
