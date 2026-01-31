
import sys
import os
from pathlib import Path
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import lifespan, state, app

async def test_integration():
    print("🧪 Testing API Integration for Recruitment Dataset...")
    
    # Evaluate the lifespan context manager
    async with lifespan(app):
        # Check if recruitment data is loaded
        if "recruitment" in state.data:
            print("✅ Recruitment data loaded successfully!")
            data = state.data["recruitment"]
            print(f"   - Train samples: {len(data['y_train'])}")
            print(f"   - Test samples: {len(data['y_test'])}")
            print(f"   - Features: {len(data['feature_names'])}")
        else:
            print("❌ Recruitment data NOT loaded!")
            
        # Check if recruitment models are loaded
        if "recruitment" in state.models:
            models = state.models["recruitment"]
            print(f"✅ Recruitment models loaded successfully! Found: {list(models.keys())}")
            
            if "xgboost" in models and "random_forest" in models and "logistic_regression" in models:
                 print("   - All expected models present.")
            else:
                 print("   - ⚠️ Some models missing.")
        else:
             print("❌ Recruitment models NOT loaded!")

        # Simulate a partial API request state
        state.active_dataset = "recruitment"
        state.active_model_id = "xgboost"
        active_model = state.get_active_model()
        
        if active_model:
            print("✅ Context switching works: Active model retrieved for recruitment dataset.")
        else:
            print("❌ Failed to retrieve active model after switching.")

if __name__ == "__main__":
    asyncio.run(test_integration())
