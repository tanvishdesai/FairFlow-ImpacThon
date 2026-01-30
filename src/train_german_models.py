
import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils.data_loader import load_german_credit_data

def train_german_models():
    print("🚀 Starting model training for German Credit Dataset...")
    
    # Load data
    data = load_german_credit_data()
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    models_dir = "models/german_credit"
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
    print(f"✅ XGBoost Accuracy: {xgb_acc:.4f}")
    joblib.dump(xgb_model, f"{models_dir}/xgboost_model.pkl")
    
    # 2. Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"✅ Random Forest Accuracy: {rf_acc:.4f}")
    joblib.dump(rf_model, f"{models_dir}/rf_model.pkl")
    
    # 3. Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
    print(f"✅ Logistic Regression Accuracy: {lr_acc:.4f}")
    joblib.dump(lr_model, f"{models_dir}/lr_model.pkl")
    
    print(f"\n🎉 All models saved to {models_dir}")

if __name__ == "__main__":
    train_german_models()
