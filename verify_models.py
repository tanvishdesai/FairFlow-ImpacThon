import requests
import json

BASE_URL = "http://localhost:8000"

def test_get_models():
    print("\n🔍 Testing GET /api/models...")
    try:
        response = requests.get(f"{BASE_URL}/api/models")
        if response.status_code == 200:
            models = response.json()
            print("✅ Successfully fetched models:")
            print(json.dumps(models, indent=2))
            return models
        else:
            print(f"❌ Failed to fetch models. Status: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return None

def test_switch_model(model_id):
    print(f"\n🔄 Testing POST /api/models/switch?model_id={model_id}...")
    try:
        response = requests.post(f"{BASE_URL}/api/models/switch?model_id={model_id}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: {result['status']}")
            print(f"✅ Active model: {result['name']}")
            return True
        elif response.status_code == 400:
            print(f"❌ Bad Request: {response.text}")
            return False
        else:
            print(f"❌ Failed to switch model. Status: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Verifying Model Switching Integration")
    
    # Check initial state
    models = test_get_models()
    
    if models:
        # Try switching to Random Forest
        if test_switch_model("random_forest"):
            # Verify it's active
            test_get_models()
            
            # Switch back to XGBoost
            test_switch_model("xgboost")
            
            # Verify again
            test_get_models()

            # Try switching to Logistic Regression
            if test_switch_model("logistic_regression"):
                 # Verify it's active
                 test_get_models()
 
                 # Switch back to XGBoost
                 test_switch_model("xgboost")

                 # Verify again
                 test_get_models()
        else:
            print("⚠️ Skipping switch back as switch failed.")
