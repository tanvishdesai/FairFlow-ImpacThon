import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Create database engine
# Check if running in a writable environment, otherwise use in-memory for safety/testing
DB_URL = "sqlite:///./fairflow.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, index=True)
    
    # Inputs
    features_json = Column(Text)  # Stored as JSON string
    
    # Outputs
    base_prediction = Column(Integer)
    base_probability = Column(Float)
    final_decision = Column(Integer)
    intervention_type = Column(String)
    
    # Analysis
    intervened = Column(Boolean)
    protected_value = Column(Integer)
    true_label = Column(Integer, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "features": json.loads(self.features_json) if self.features_json else [],
            "base_prediction": self.base_prediction,
            "base_probability": self.base_probability,
            "final_decision": self.final_decision,
            "intervention_type": self.intervention_type,
            "intervened": self.intervened,
            "protected_value": self.protected_value,
            "true_label": self.true_label
        }

class AuditLog(Base):
    """
    Separate table for audit logging if we want to separate raw predictions from audit trails.
    For now, this might be redundant with Prediction, but good for separation of concerns
    if the audit log needs to store more meta-data or be immutable.
    
    Currently, the system uses 'audit_log' and 'predictions' very similarly. 
    we'll map the frontend 'AuditLogEntry' to this.
    """
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, index=True) # Link to original prediction if needed
    timestamp = Column(String)
    
    base_prediction = Column(Integer)
    final_decision = Column(Integer)
    intervention_type = Column(String)
    protected_value = Column(Integer)
    true_label = Column(Integer, nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "base_prediction": self.base_prediction,
            "final_decision": self.final_decision,
            "intervention_type": self.intervention_type,
            "protected_value": self.protected_value,
            "true_label": self.true_label
        }

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
