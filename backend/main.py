"""
FairFlow Backend - FastAPI Application

This is the main API server for the FairFlow bias mitigation platform.
It provides endpoints for predictions, fairness metrics, and explanations.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import asyncio
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_adult_data, load_german_credit_data
from src.utils.metrics import calculate_all_metrics, calculate_demographic_parity
from src.explainability.shap_explainer import ShapExplainer
from backend.database import init_db, SessionLocal, Prediction, AuditLog
import json



# ============================================================================
# Pydantic Models
# ============================================================================

class ApplicantData(BaseModel):
    """Input data for a single applicant/prediction request."""
    features: Dict[str, Any] = Field(..., description="Feature name to value mapping")


class PredictionResponse(BaseModel):
    """Response for a prediction request."""
    id: int
    timestamp: str
    base_prediction: int
    base_probability: float
    fairflow_decision: int
    intervened: bool
    intervention_type: Optional[str]


class MetricsResponse(BaseModel):
    """Current fairness and performance metrics."""
    timestamp: str
    accuracy: float
    demographic_parity_ratio: float
    demographic_parity_difference: float
    privileged_approval_rate: float
    unprivileged_approval_rate: float
    is_fair: bool
    total_predictions: int
    total_interventions: int
    intervention_rate: float


class AuditLogEntry(BaseModel):
    """Single entry in the audit log."""
    id: int
    timestamp: str
    base_prediction: int
    final_decision: int
    intervention_type: str
    protected_value: int
    true_label: Optional[int]


class ExplanationResponse(BaseModel):
    """Response for an explanation request."""
    id: int
    prediction: int
    probability_approve: float
    intervention_type: str
    intervention_reason: str
    detailed_reason: str
    top_contributors: List[Dict[str, Any]]
    waterfall_plot: str


class SimulationStatus(BaseModel):
    """Status of the simulation."""
    is_running: bool
    samples_processed: int
    current_accuracy: float
    current_dpr: float
    fairflow_active: bool


class FairFlowConfig(BaseModel):
    """Configuration for universal FairFlow."""
    protected_attribute: Optional[str] = Field(default="sex", description="Name of protected attribute")
    fairness_threshold: float = Field(default=0.8, description="Minimum acceptable DPR")
    fairness_metric: str = Field(default="demographic_parity", description="Fairness metric to optimize")
    accuracy_weight: float = Field(default=0.4, description="Weight for accuracy in trade-off")
    fairness_weight: float = Field(default=0.6, description="Weight for fairness in trade-off")
    use_universal_agent: bool = Field(default=True, description="Use universal (dataset-agnostic) agent")


# ============================================================================
# Application State
# ============================================================================

class DatasetSwitchRequest(BaseModel):
    dataset_id: str

class ModelSwitchRequest(BaseModel):
    model_id: str


class AppState:
    """Global application state."""
    
    def __init__(self):
        self.models = {}  # {dataset_id: {model_id: model}}
        self.data = {}    # {dataset_id: data_dict}
        self.active_dataset = "adult"
        self.active_model_id = "xgboost"
        
        # Universal RL Agent (Shared across datasets)
        self.universal_rl_agent = None 
        # Dataset-specific RL agent (Deprecating, but keeping for compatibility if needed)
        self.rl_agent = None
        
        self.explainer = None
        
        # Configuration
        self.config = {
            "protected_attribute": "sex",
            "fairness_threshold": 0.8,
            "fairness_metric": "demographic_parity",
            "accuracy_weight": 0.4,
            "fairness_weight": 0.6,
            "use_universal_agent": True
        }
        
        # Simulation state
        self.simulation_running = False
        self.fairflow_active = True
        self.simulation_task = None
        
        # Drift injection state
        self.drift_active = False
        self.drift_samples_remaining = 0
        self.drift_unprivileged_ratio = 0.9
        
        # Prediction history
        self.predictions = []
        self.audit_log = []
        self.next_id = 1
        
        # Rolling metrics
        self.decisions_window = []
        self.window_size = 100
        
        # Universal agent statistics
        self.privileged_decisions = []
        self.unprivileged_decisions = []
        self.privileged_confidences = []
        self.unprivileged_confidences = []

    def get_active_model(self):
        """Get the currently active base model."""
        if self.active_dataset in self.models and self.active_model_id in self.models[self.active_dataset]:
            return self.models[self.active_dataset][self.active_model_id]
        return None
        
    def get_active_data(self):
        """Get the currently active dataset."""
        return self.data.get(self.active_dataset)

    def get_feature_names(self):
        data = self.get_active_data()
        return data["feature_names"] if data else []

    def get_label_encoders(self):
        data = self.get_active_data()
        return data.get("label_encoders", {}) if data else {}

    def get_scaler(self):
        data = self.get_active_data()
        return data.get("scaler") if data else None

state = AppState()


# ============================================================================
# Startup/Shutdown
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("🚀 Starting FairFlow API server...")
    
    init_db()
    
    base_dir = Path(__file__).parent.parent
    
    # 1. Load Adult Census Data & Models
    print("📥 Loading Adult Census Data...")
    try:
        adult_data = load_adult_data(data_dir=str(base_dir / "data"), protected_attribute="sex")
        state.data["adult"] = adult_data
        
        state.models["adult"] = {}
        # Load Adult models
        try:
            state.models["adult"]["xgboost"] = joblib.load(base_dir / "models/base_model/xgboost_biased.joblib")
            state.models["adult"]["random_forest"] = joblib.load(base_dir / "models/rf_model/random_forest_biased.joblib")
            state.models["adult"]["logistic_regression"] = joblib.load(base_dir / "models/lr_model/logistic_regression_biased.joblib")
        except Exception as e:
            print(f"⚠️ Partial failure loading Adult models: {e}")
            
    except Exception as e:
        print(f"⚠️ Could not load Adult data: {e}")

    # 2. Load German Credit Data & Models
    print("📥 Loading German Credit Data...")
    try:
        german_data = load_german_credit_data(data_dir=str(base_dir / "data"), protected_attribute="Sex")
        state.data["german"] = german_data
        
        state.models["german"] = {}
        # Load German models
        try:
            state.models["german"]["xgboost"] = joblib.load(base_dir / "models/german_credit/xgboost_model.pkl")
            state.models["german"]["random_forest"] = joblib.load(base_dir / "models/german_credit/rf_model.pkl")
            state.models["german"]["logistic_regression"] = joblib.load(base_dir / "models/german_credit/lr_model.pkl")
        except Exception as e:
            print(f"⚠️ Partial failure loading German models: {e}")
            
    except Exception as e:
        print(f"⚠️ Could not load German Credit data: {e}")
        
    # 3. Load Universal Agent
    print("🌍 Loading Universal RL Agent...")
    universal_agent_path = base_dir / "models/rl_agent/ppo_universal_fairness_agent.zip"
    from stable_baselines3 import PPO
    
    if universal_agent_path.exists():
        state.universal_rl_agent = PPO.load(str(universal_agent_path))
        print("✅ Universal RL agent loaded and ready for ALL datasets!")
    else:
        print("⚠️ Universal agent not found.")

    print("🎉 FairFlow API ready!")
    
    yield
    
    print("👋 Shutting down FairFlow API...")
    state.simulation_running = False


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="FairFlow API",
    description="RL-Driven Adaptive Bias Firewall for Fair AI Decision Making",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def sanitize_float(value: float, default: float = 0.0) -> float:
    """Convert NaN or Inf values to JSON-safe defaults."""
    import math
    if math.isnan(value):
        return default
    if math.isinf(value):
        return 1e10 if value > 0 else -1e10
    return value


def get_universal_state(base_pred: int, base_prob: float, protected_value: int) -> np.ndarray:
    """
    Build the 12-dimensional universal state vector for the dataset-agnostic agent.
    """
    # Calculate rolling metrics
    dpr = calculate_current_dpr()
    
    # ... (Rest of logic is state-dependent but generic)
    # Using existing state lists which are populated by predictions regardless of dataset
    # This assumes we want the agent to adapt to the CURRENT stream of data
    
    priv_approval = np.mean(state.privileged_decisions[-50:]) if state.privileged_decisions else 0.5
    unpriv_approval = np.mean(state.unprivileged_decisions[-50:]) if state.unprivileged_decisions else 0.5
    
    # Intervention rate
    total_preds = len(state.predictions)
    # ... (keeping existing logic)
    total_interventions = sum(1 for p in state.predictions[-100:] if p["base_prediction"] != p["final_decision"]) # Check last 100 for relevance
    intervention_rate = total_interventions / min(len(state.predictions[-100:]), 100) if state.predictions else 0.0
    
    # ... (Rest is fine)
    
    # Group ratio
    n_unpriv = len(state.unprivileged_decisions)
    n_total = len(state.privileged_decisions) + n_unpriv
    group_ratio = n_unpriv / max(n_total, 1)
    
    # Confidence gap
    priv_conf = np.mean(state.privileged_confidences[-50:]) if state.privileged_confidences else 0.5
    unpriv_conf = np.mean(state.unprivileged_confidences[-50:]) if state.unprivileged_confidences else 0.5
    confidence_gap = priv_conf - unpriv_conf
    
    # Normalize DPR and differences to [0, 1]
    def normalize_dpr(dpr): return min(max(dpr / 2.0, 0.0), 1.0)
    def normalize_diff(diff): return min(max((diff + 1.0) / 2.0, 0.0), 1.0)
    
    universal_state = np.array([
        float(base_pred),              # 0: Base prediction
        base_prob,                     # 1: Base confidence
        float(protected_value),        # 2: Protected value
        normalize_dpr(dpr),            # 3: DPR (normalized)
        0.5,                           # 4: TPR diff (placeholder)
        0.5,                           # 5: FPR diff (placeholder)
        priv_approval,                 # 6: Privileged approval rate
        unpriv_approval,               # 7: Unprivileged approval rate
        min(intervention_rate, 1.0),   # 8: Intervention rate
        group_ratio,                   # 9: Group ratio
        0.5,                           # 10: Consecutive same-group (placeholder)
        normalize_diff(confidence_gap) # 11: Confidence gap
    ], dtype=np.float32)
    
    return universal_state


def get_fairflow_decision(features: np.ndarray, base_pred: int, base_prob: float, protected_value: int = 0) -> tuple:
    """
    Get FairFlow's decision using RL agent or rule-based fallback.
    
    Supports both universal (dataset-agnostic) and dataset-specific agents.
    
    Returns:
        (final_decision, intervention_type)
    """
    if not state.fairflow_active:
        return base_pred, "FAIRFLOW_DISABLED"
    
    # Try universal agent first (if configured and available)
    if state.config.get("use_universal_agent", True) and state.universal_rl_agent is not None:
        # Use UNIVERSAL agent with 12-dimensional state
        obs = get_universal_state(base_pred, base_prob, protected_value)
        action, _ = state.universal_rl_agent.predict(obs, deterministic=True)
        
        if action == 0:  # APPROVE (use base)
            return base_pred, "ACCEPTED"
        elif action == 1:  # DENY
            return 0, "OVERRIDE_TO_DENY" if base_pred != 0 else "ACCEPTED"
        else:  # ACCEPT (force approve)
            return 1, "OVERRIDE_TO_APPROVE" if base_pred != 1 else "ACCEPTED"
    
    elif state.rl_agent is not None:
        # Use dataset-specific RL agent (legacy mode)
        current_dpr = calculate_current_dpr()
        obs = np.concatenate([[base_pred, base_prob], features.flatten(), [current_dpr]])
        action, _ = state.rl_agent.predict(obs.astype(np.float32), deterministic=True)
        
        if action == 0:  # APPROVE
            return 1, "OVERRIDE_TO_APPROVE" if base_pred != 1 else "ACCEPTED"
        elif action == 1:  # DENY
            return 0, "OVERRIDE_TO_DENY" if base_pred != 0 else "ACCEPTED"
        else:  # ACCEPT
            return base_pred, "ACCEPTED"
    else:
        # Rule-based fallback
        current_dpr = calculate_current_dpr()
        threshold = state.config.get("fairness_threshold", 0.8)
        
        # If DPR is too low and base model is denying unprivileged, consider approving
        if current_dpr < threshold and base_pred == 0 and protected_value == 0 and base_prob > 0.35:
            return 1, "OVERRIDE_TO_APPROVE"
        
        return base_pred, "ACCEPTED"


def calculate_current_dpr() -> float:
    """Calculate current demographic parity ratio from recent decisions."""
    if len(state.decisions_window) < 10:
        return 1.0
    
    decisions = np.array([d["decision"] for d in state.decisions_window])
    protected = np.array([d["protected"] for d in state.decisions_window])
    
    result = calculate_demographic_parity(decisions, protected)
    return result["demographic_parity_ratio"]


def calculate_rolling_metrics() -> dict:
    """Calculate rolling metrics from recent decisions."""
    if len(state.decisions_window) == 0:
        return {
            "accuracy": 0.0,
            "dpr": 1.0,
            "intervention_rate": 0.0
        }
    
    decisions = np.array([d["decision"] for d in state.decisions_window])
    protected = np.array([d["protected"] for d in state.decisions_window])
    true_labels = np.array([d.get("true_label", d["decision"]) for d in state.decisions_window])
    interventions = np.array([d["intervened"] for d in state.decisions_window])
    
    accuracy = np.mean(decisions == true_labels)
    dpr_result = calculate_demographic_parity(decisions, protected)
    intervention_rate = np.mean(interventions)
    
    return {
        "accuracy": accuracy,
        "dpr": dpr_result["demographic_parity_ratio"],
        "intervention_rate": intervention_rate,
        **dpr_result
    }


# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "name": "FairFlow API",
        "version": "1.0.0",
        "status": "running",
        "fairflow_active": state.fairflow_active,
        "models_loaded": {
            "base_model": state.get_active_model() is not None,
            "rl_agent": state.rl_agent is not None,
            "explainer": state.explainer is not None
        }
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(applicant: ApplicantData):
    """
    Make a prediction for a single applicant.
    
    The request contains feature values, and the response includes
    both the base model prediction and FairFlow's final decision.
    """
    base_model = state.get_active_model()
    if base_model is None:
        raise HTTPException(status_code=503, detail="Base model not loaded")
    
    feature_names = state.get_feature_names()
    label_encoders = state.get_label_encoders()
    
    # Process features: handle both numeric and categorical strings
    processed_features = []
    
    for f in feature_names:
        raw_val = applicant.features.get(f)
        
        # Default to 0/mode if missing
        if raw_val is None:
            processed_features.append(0.0)
            continue
            
        if f in label_encoders:
            # Need to encode string -> int
            le = label_encoders[f]
            try:
                # Handle unknown labels gracefully-ish (e.g. use first class)
                if str(raw_val) in le.classes_:
                    encoded_val = le.transform([str(raw_val)])[0]
                else:
                    # Fallback for unknown category
                    print(f"⚠️ Warning: Unknown category '{raw_val}' for feature '{f}'. Using default.")
                    encoded_val = 0
                processed_features.append(float(encoded_val))
            except Exception as e:
                print(f"❌ Encoding error for {f}: {e}")
                processed_features.append(0.0)
        else:
            # Numeric feature
            try:
                processed_features.append(float(raw_val))
            except (ValueError, TypeError):
                print(f"❌ Value error for {f}: {raw_val}")
                processed_features.append(0.0)
                
    features = np.array(processed_features).reshape(1, -1)
    
    # Scale features if scaler exists
    scaler = state.get_scaler()
    if scaler:
        try:
             features = scaler.transform(pd.DataFrame(features, columns=feature_names))
        except Exception as e:
            print(f"⚠️ Scaling failed: {e}")
            
    # Get base model prediction
    base_pred = int(base_model.predict(features)[0])
    base_prob = float(base_model.predict_proba(features)[0, 1])
    
    # Get FairFlow decision
    final_decision, intervention_type = get_fairflow_decision(features, base_pred, base_prob)
    intervened = final_decision != base_pred
    
    # Generate response
    prediction_id = state.next_id
    state.next_id += 1
    timestamp = datetime.now().isoformat()
    
    # Add to history
    state.predictions.append({
        "id": prediction_id,
        "timestamp": timestamp,
        "features": features[0].tolist(),
        "base_prediction": base_pred,
        "final_decision": final_decision,
        "intervention_type": intervention_type
    })
    
    # Add to decisions window (assume protected = 0 for API calls without protected info)
    state.decisions_window.append({
        "decision": final_decision,
        "protected": 0,  # Default, would be extracted from features in production
        "intervened": intervened,
        "true_label": final_decision  # Unknown ground truth
    })
    if len(state.decisions_window) > state.window_size:
        state.decisions_window.pop(0)
    
    # Add to audit log (In-memory - optional, keep for now if needed by other components, but DB is primary)
    state.audit_log.append({
        "id": prediction_id,
        "timestamp": timestamp,
        "base_prediction": base_pred,
        "final_decision": final_decision,
        "intervention_type": intervention_type,
        "protected_value": 0,
        "true_label": None
    })
    
    # Persist to Database
    db = SessionLocal()
    try:
        db_prediction = Prediction(
            timestamp=timestamp,
            features_json=json.dumps(features[0].tolist()),
            base_prediction=base_pred,
            base_probability=base_prob,
            final_decision=final_decision,
            intervention_type=intervention_type if intervention_type else "None",  # Handle None
            intervened=intervened,
            protected_value=0, # Default for API
            true_label=None
        )
        db_audit = AuditLog(
            timestamp=timestamp,
            base_prediction=base_pred,
            final_decision=final_decision,
            intervention_type=intervention_type if intervention_type else "None",
            protected_value=0,
            true_label=None
        )
        db.add(db_prediction)
        db.add(db_audit)
        db.commit()
    finally:
        db.close()

    
    return PredictionResponse(
        id=prediction_id,
        timestamp=timestamp,
        base_prediction=base_pred,
        base_probability=base_prob,
        fairflow_decision=final_decision,
        intervened=intervened,
        intervention_type=intervention_type if intervened else None
    )


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get current fairness and performance metrics."""
    metrics = calculate_rolling_metrics()
    
    total_predictions = len(state.predictions)
    total_interventions = sum(
        1 for p in state.predictions 
        if p["base_prediction"] != p["final_decision"]
    )
    
    return MetricsResponse(
        timestamp=datetime.now().isoformat(),
        accuracy=sanitize_float(metrics.get("accuracy", 0.0)),
        demographic_parity_ratio=sanitize_float(metrics.get("dpr", 1.0), default=1.0),
        demographic_parity_difference=sanitize_float(metrics.get("demographic_parity_difference", 0.0)),
        privileged_approval_rate=sanitize_float(metrics.get("privileged_approval_rate", 0.0)),
        unprivileged_approval_rate=sanitize_float(metrics.get("unprivileged_approval_rate", 0.0)),
        is_fair=metrics.get("is_fair", True),
        total_predictions=total_predictions,
        total_interventions=total_interventions,
        intervention_rate=total_interventions / max(1, total_predictions)
    )


@app.get("/api/audit-log", response_model=List[AuditLogEntry])
async def get_audit_log(limit: int = 50):
    """Get recent entries from the audit log."""
    # return [AuditLogEntry(**entry) for entry in state.audit_log[-limit:]]
    
    db = SessionLocal()
    try:
        logs = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit).all()
        return [
            AuditLogEntry(
                id=log.id, 
                timestamp=log.timestamp,
                base_prediction=log.base_prediction,
                final_decision=log.final_decision,
                intervention_type=log.intervention_type,
                protected_value=log.protected_value,
                true_label=log.true_label
            ) for log in logs
        ]
    finally:
        db.close()



@app.get("/api/explain/{prediction_id}", response_model=ExplanationResponse)
async def get_explanation(prediction_id: int):
    """Get SHAP explanation for a specific prediction."""
    if state.explainer is None:
        raise HTTPException(status_code=503, detail="Explainer not loaded")
    
    # Find the prediction
    prediction = None
    for p in state.predictions:
        if p["id"] == prediction_id:
            prediction = p
            break
    
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Generate explanation
    features = np.array(prediction["features"]).reshape(1, -1)
    
    explanation = state.explainer.generate_intervention_explanation(
        X=features,
        base_prediction=prediction["base_prediction"],
        final_decision=prediction["final_decision"],
        sample_idx=prediction_id
    )
    
    return ExplanationResponse(
        id=prediction_id,
        prediction=explanation["prediction"],
        probability_approve=explanation["probability_approve"],
        intervention_type=explanation["intervention_type"],
        intervention_reason=explanation["intervention_reason"],
        detailed_reason=explanation["detailed_reason"],
        top_contributors=explanation["contributions"][:5],
        waterfall_plot=explanation["waterfall_plot"]
    )


@app.post("/api/fairflow/toggle")
async def toggle_fairflow(active: bool):
    """Enable or disable FairFlow interventions."""
    state.fairflow_active = active
    return {"fairflow_active": state.fairflow_active}


@app.get("/api/datasets")
async def get_datasets():
    """Get available datasets."""
    return [
        {"id": "adult", "name": "Adult Census (Income)", "active": state.active_dataset == "adult"},
        {"id": "german", "name": "German Credit (Risk)", "active": state.active_dataset == "german"}
    ]

@app.post("/api/dataset/switch")
async def switch_dataset(dataset: DatasetSwitchRequest):
    """Switch the active dataset."""
    if dataset.dataset_id not in state.data:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    state.active_dataset = dataset.dataset_id
    # Reset model to default for the new dataset
    state.active_model_id = "xgboost" 
    
    print(f"🔄 Switched to dataset: {dataset.dataset_id}")
    return {"status": "success", "dataset": dataset.dataset_id}

@app.get("/api/models")
async def get_models():
    """Get available models for the CURRENT dataset."""
    if state.active_dataset not in state.models:
        return []
        
    current_models = state.models[state.active_dataset]
    return [
        {"id": m_id, "name": m_id.replace("_", " ").title(), "active": m_id == state.active_model_id}
        for m_id in current_models.keys()
    ]

@app.post("/api/models/switch")
async def switch_model(model: ModelSwitchRequest):
    """Switch the active model for the current dataset."""
    current_models = state.models.get(state.active_dataset, {})
    if model.model_id not in current_models:
        raise HTTPException(status_code=404, detail="Model not found for current dataset")
    
    state.active_model_id = model.model_id
    print(f"🔄 Switched to model: {model.model_id}")
    return {"status": "success", "model": model.model_id}

@app.get("/api/fairflow/status")
async def get_fairflow_status():
    """Get current FairFlow status."""
    return {
        "active": state.fairflow_active,
        "universal_agent_loaded": state.universal_rl_agent is not None,
        "mode": "universal_rl_agent" if state.universal_rl_agent else "rule_based",
        "config": state.config,
        "active_dataset": state.active_dataset,  # Expose active dataset
        "active_model": state.active_model_id
    }


# ============================================================================
# Simulation Endpoints
# ============================================================================

@app.post("/api/simulate/start")
async def start_simulation(background_tasks: BackgroundTasks, speed: float = 1.0):
    """Start simulating predictions from test data."""
    if state.simulation_running:
        return {"status": "already_running"}
    
    if state.data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    state.simulation_running = True
    background_tasks.add_task(run_simulation, speed)
    
    return {"status": "started", "speed": speed}


@app.post("/api/simulate/stop")
async def stop_simulation():
    """Stop the simulation."""
    state.simulation_running = False
    return {"status": "stopped"}


@app.get("/api/simulate/status", response_model=SimulationStatus)
async def get_simulation_status():
    """Get current simulation status."""
    metrics = calculate_rolling_metrics()
    
    return SimulationStatus(
        is_running=state.simulation_running,
        samples_processed=len(state.predictions),
        current_accuracy=sanitize_float(metrics.get("accuracy", 0.0)),
        current_dpr=sanitize_float(metrics.get("dpr", 1.0), default=1.0),
        fairflow_active=state.fairflow_active
    )


@app.post("/api/simulate/inject-drift")
async def inject_drift(unprivileged_ratio: float = 0.9, duration: int = 50):
    """
    Inject biased data into the simulation.
    
    This simulates a scenario where more unprivileged (female) applicants
    start applying, exposing the base model's gender bias more clearly.
    
    Args:
        unprivileged_ratio: Ratio of female applicants (0.0-1.0), default 0.9
        duration: Number of samples to apply drift for, default 50
    """
    state.drift_active = True
    state.drift_samples_remaining = duration
    state.drift_unprivileged_ratio = unprivileged_ratio
    
    return {
        "status": "drift_injected",
        "unprivileged_ratio": unprivileged_ratio,
        "duration": duration,
        "message": f"Injecting {int(unprivileged_ratio*100)}% female applicants for next {duration} samples"
    }


async def run_simulation(speed: float = 1.0):
    """Background task to run simulation."""
    data = state.get_active_data()
    base_model = state.get_active_model()
    
    if data is None or base_model is None:
        return
    
    X_test = data["X_test"].values
    y_test = data["y_test"].values
    protected_test = data["protected_test"].values
    
    n_samples = len(X_test)
    
    # Create indices for privileged (male=1) and unprivileged (female=0) groups
    privileged_indices = np.where(protected_test == 1)[0]
    unprivileged_indices = np.where(protected_test == 0)[0]
    
    # Shuffle both sets
    np.random.shuffle(privileged_indices)
    np.random.shuffle(unprivileged_indices)
    
    priv_ptr = 0
    unpriv_ptr = 0
    samples_processed = 0
    
    while state.simulation_running and samples_processed < n_samples:
        # Determine which group to sample from
        if state.drift_active and state.drift_samples_remaining > 0:
            # During drift: heavily favor unprivileged (female) samples
            use_unprivileged = np.random.random() < state.drift_unprivileged_ratio
            state.drift_samples_remaining -= 1
            if state.drift_samples_remaining == 0:
                state.drift_active = False
        else:
            # Normal operation: roughly 50/50 or based on natural distribution
            use_unprivileged = np.random.random() < 0.5
        
        # Get the next sample from appropriate group
        if use_unprivileged and unpriv_ptr < len(unprivileged_indices):
            idx = unprivileged_indices[unpriv_ptr]
            unpriv_ptr += 1
        elif priv_ptr < len(privileged_indices):
            idx = privileged_indices[priv_ptr]
            priv_ptr += 1
        elif unpriv_ptr < len(unprivileged_indices):
            idx = unprivileged_indices[unpriv_ptr]
            unpriv_ptr += 1
        else:
            break  # No more samples
        
        features = X_test[idx].reshape(1, -1)
        true_label = int(y_test[idx])
        protected_val = int(protected_test[idx])
        
        # Get base prediction
        base_pred = int(base_model.predict(features)[0])
        base_prob = float(base_model.predict_proba(features)[0, 1])
        
        # Get FairFlow decision (pass protected_value for universal agent)
        final_decision, intervention_type = get_fairflow_decision(
            features, base_pred, base_prob, protected_val
        )
        intervened = final_decision != base_pred
        
        # Update universal agent statistics for state building
        if protected_val == 1:
            state.privileged_decisions.append(final_decision)
            state.privileged_confidences.append(base_prob)
        else:
            state.unprivileged_decisions.append(final_decision)
            state.unprivileged_confidences.append(base_prob)
        
        # Keep rolling window size manageable
        if len(state.privileged_decisions) > 100:
            state.privileged_decisions = state.privileged_decisions[-100:]
            state.privileged_confidences = state.privileged_confidences[-100:]
        if len(state.unprivileged_decisions) > 100:
            state.unprivileged_decisions = state.unprivileged_decisions[-100:]
            state.unprivileged_confidences = state.unprivileged_confidences[-100:]
        
        # Record
        prediction_id = state.next_id
        state.next_id += 1
        timestamp = datetime.now().isoformat()
        
        state.predictions.append({
            "id": prediction_id,
            "timestamp": timestamp,
            "features": features[0].tolist(),
            "base_prediction": base_pred,
            "final_decision": final_decision,
            "intervention_type": intervention_type
        })
        
        state.decisions_window.append({
            "decision": final_decision,
            "protected": protected_val,
            "intervened": intervened,
            "true_label": true_label
        })
        if len(state.decisions_window) > state.window_size:
            state.decisions_window.pop(0)
        
        state.audit_log.append({
            "id": prediction_id,
            "timestamp": timestamp,
            "base_prediction": base_pred,
            "final_decision": final_decision,
            "intervention_type": intervention_type,
            "protected_value": protected_val,
            "true_label": true_label
        })
        
        # Persist to Database
        db = SessionLocal()
        try:
            db_prediction = Prediction(
                timestamp=timestamp,
                features_json=json.dumps(features[0].tolist()),
                base_prediction=base_pred,
                base_probability=base_prob,
                final_decision=final_decision,
                intervention_type=intervention_type if intervention_type else "None",
                intervened=intervened,
                protected_value=protected_val,
                true_label=true_label
            )
            db_audit = AuditLog(
                timestamp=timestamp,
                base_prediction=base_pred,
                final_decision=final_decision,
                intervention_type=intervention_type if intervention_type else "None",
                protected_value=protected_val,
                true_label=true_label
            )
            db.add(db_prediction)
            db.add(db_audit)
            db.commit()
        finally:
            db.close()
        
        samples_processed += 1
        await asyncio.sleep(0.5 / speed)  # Delay between samples
    
    state.simulation_running = False



# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
