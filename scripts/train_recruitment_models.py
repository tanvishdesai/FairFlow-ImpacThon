
import os
import sys
from pathlib import Path
import joblib

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_fair_recruitment_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models():
    print("Loading Fair Recruitment data...")
    try:
        data = load_fair_recruitment_data(
            data_dir=r"c:\Users\DELL\Desktop\hckton\ImpactThon\fairflow\data", # Use absolute path
            protected_attribute="Gender"
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    X_train = data["X_train"]
    y_train = data["y_train"] 
    
    # Ensure y is 1D array
    y_train = y_train.values.ravel()

    models_dir = Path(r"c:\Users\DELL\Desktop\hckton\ImpactThon\fairflow\models\recruitment")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training on {X_train.shape[0]} samples...")

    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, models_dir / "lr_model.pkl")
    print(f"✅ Saved Logistic Regression to {models_dir / 'lr_model.pkl'}")

    # 2. Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, models_dir / "rf_model.pkl")
    print(f"✅ Saved Random Forest to {models_dir / 'rf_model.pkl'}")

    # 3. XGBoost
    print("Training XGBoost...")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, models_dir / "xgboost_model.pkl")
    print(f"✅ Saved XGBoost to {models_dir / 'xgboost_model.pkl'}")

if __name__ == "__main__":
    train_models()
