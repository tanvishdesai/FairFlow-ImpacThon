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

from src.utils.data_loader import load_adult_data
from src.utils.metrics import calculate_all_metrics, calculate_demographic_parity
from src.explainability.shap_explainer import ShapExplainer


# ============================================================================
# Pydantic Models
# ============================================================================

class ApplicantData(BaseModel):
    """Input data for a single applicant/prediction request."""
    features: Dict[str, float] = Field(..., description="Feature name to value mapping")


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

class AppState:
    """Global application state."""
    
    def __init__(self):
        self.base_model = None
        self.rl_agent = None
        self.universal_rl_agent = None  # NEW: Universal (dataset-agnostic) agent
        self.fairflow_wrapper = None    # NEW: Universal wrapper
        self.explainer = None
        self.data = None
        self.feature_names = []
        self.scaler = None
        
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
        self.drift_unprivileged_ratio = 0.9  # 90% female when drift is active
        
        # Prediction history
        self.predictions = []
        self.audit_log = []
        self.next_id = 1
        
        # Rolling metrics
        self.decisions_window = []
        self.window_size = 100
        
        # Universal agent statistics (for feature-agnostic state)
        self.privileged_decisions = []
        self.unprivileged_decisions = []
        self.privileged_confidences = []
        self.unprivileged_confidences = []


state = AppState()


# ============================================================================
# Startup/Shutdown
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 Starting FairFlow API server...")
    
    base_dir = Path(__file__).parent.parent
    
    # Load data
    print("📥 Loading data...")
    try:
        state.data = load_adult_data(data_dir=str(base_dir / "data"), protected_attribute="sex")
        state.feature_names = state.data["feature_names"]
        state.scaler = state.data.get("scaler")
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load data: {e}")
    
    # Load base model
    model_path = base_dir / "models" / "base_model" / "xgboost_biased.joblib"
    if model_path.exists():
        print(f"📦 Loading base model from {model_path}...")
        state.base_model = joblib.load(model_path)
        state.explainer = ShapExplainer(state.base_model, state.feature_names)
        print("✅ Base model and explainer loaded")
    else:
        print("⚠️ Base model not found. Train it first with: python src/train_base_model.py")
    
    # Load RL agents (try universal first, then dataset-specific)
    universal_agent_path = base_dir / "models" / "rl_agent" / "ppo_universal_fairness_agent.zip"
    dataset_agent_path = base_dir / "models" / "rl_agent" / "ppo_fairness_agent.zip"
    
    from stable_baselines3 import PPO
    
    # Try universal agent first (preferred)
    if universal_agent_path.exists():
        print(f"🌍 Loading UNIVERSAL RL agent from {universal_agent_path}...")
        state.universal_rl_agent = PPO.load(str(universal_agent_path))
        print("✅ Universal RL agent loaded (dataset-agnostic)")
    
    # Also load dataset-specific agent if available
    if dataset_agent_path.exists():
        print(f"🤖 Loading dataset-specific RL agent from {dataset_agent_path}...")
        state.rl_agent = PPO.load(str(dataset_agent_path))
        print("✅ Dataset-specific RL agent loaded")
    
    if state.universal_rl_agent is None and state.rl_agent is None:
        print("⚠️ No RL agent found. Using rule-based fallback.")
        print("   Train universal agent with: python src/agents/train_universal.py")
    
    print("🎉 FairFlow API ready!")
    
    yield
    
    # Shutdown
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
    priv_approval = np.mean(state.privileged_decisions[-50:]) if state.privileged_decisions else 0.5
    unpriv_approval = np.mean(state.unprivileged_decisions[-50:]) if state.unprivileged_decisions else 0.5
    
    # Intervention rate
    total_preds = len(state.predictions)
    total_interventions = sum(1 for p in state.predictions if p["base_prediction"] != p["final_decision"])
    intervention_rate = total_interventions / max(total_preds, 1)
    
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
            "base_model": state.base_model is not None,
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
    if state.base_model is None:
        raise HTTPException(status_code=503, detail="Base model not loaded")
    
    # Convert features to array
    features = np.array([applicant.features.get(f, 0.0) for f in state.feature_names])
    features = features.reshape(1, -1)
    
    # Get base model prediction
    base_pred = int(state.base_model.predict(features)[0])
    base_prob = float(state.base_model.predict_proba(features)[0, 1])
    
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
    
    # Add to audit log
    state.audit_log.append({
        "id": prediction_id,
        "timestamp": timestamp,
        "base_prediction": base_pred,
        "final_decision": final_decision,
        "intervention_type": intervention_type,
        "protected_value": 0,
        "true_label": None
    })
    
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
    return [AuditLogEntry(**entry) for entry in state.audit_log[-limit:]]


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


@app.get("/api/fairflow/status")
async def get_fairflow_status():
    """Get current FairFlow status."""
    if state.universal_rl_agent is not None:
        mode = "universal_rl_agent"
    elif state.rl_agent is not None:
        mode = "dataset_specific_rl_agent"
    else:
        mode = "rule_based"
    
    return {
        "active": state.fairflow_active,
        "universal_agent_loaded": state.universal_rl_agent is not None,
        "dataset_agent_loaded": state.rl_agent is not None,
        "mode": mode,
        "config": state.config,
        "is_universal": state.config.get("use_universal_agent", True) and state.universal_rl_agent is not None
    }


@app.post("/api/fairflow/configure")
async def configure_fairflow(config: FairFlowConfig):
    """
    Configure FairFlow for a specific use case.
    
    This allows customizing:
    - Protected attribute to monitor
    - Fairness threshold
    - Accuracy vs fairness trade-off weights
    - Whether to use universal or dataset-specific agent
    """
    state.config = {
        "protected_attribute": config.protected_attribute,
        "fairness_threshold": config.fairness_threshold,
        "fairness_metric": config.fairness_metric,
        "accuracy_weight": config.accuracy_weight,
        "fairness_weight": config.fairness_weight,
        "use_universal_agent": config.use_universal_agent
    }
    
    return {
        "status": "configured",
        "config": state.config,
        "mode": "universal" if config.use_universal_agent and state.universal_rl_agent else "dataset_specific"
    }


@app.get("/api/fairflow/config")
async def get_fairflow_config():
    """Get current FairFlow configuration."""
    return state.config


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
    if state.data is None or state.base_model is None:
        return
    
    X_test = state.data["X_test"].values
    y_test = state.data["y_test"].values
    protected_test = state.data["protected_test"].values
    
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
        base_pred = int(state.base_model.predict(features)[0])
        base_prob = float(state.base_model.predict_proba(features)[0, 1])
        
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
        
        samples_processed += 1
        await asyncio.sleep(0.5 / speed)  # Delay between samples
    
    state.simulation_running = False


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
