# FairFlow: Hackathon Presentation Content

## 📋 Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Proposed Solution](#2-proposed-solution)
3. [Reinforcement Learning Layer (Deep Dive)](#3-reinforcement-learning-layer-deep-dive)
4. [Pipeline Execution Diagram Description](#4-pipeline-execution-diagram-description)
5. [Architecture Diagram Description](#5-architecture-diagram-description)

---

## 1. Problem Statement

> **Banks, insurance companies, and HR firms face regulatory nightmares as their deployed AI models become biased over time due to data drift in production—current static fairness solutions fix bias at training time but fail to maintain compliance continuously, exposing organizations to EU AI Act and GDPR violations.**

> **The critical gap: A model that was fair yesterday can become discriminatory today, and there's no real-time self-healing mechanism to dynamically balance accuracy and fairness in production environments.**

---

## 2. Proposed Solution

### FairFlow: The RL-Driven Adaptive Bias Firewall

**FairFlow** is an enterprise-grade AI Governance Platform that acts as a **"Self-Healing Bias Firewall"** for black-box machine learning models. It uses **Deep Reinforcement Learning (PPO)** to dynamically balance predictive accuracy with regulatory fairness requirements in real-time.

### Main Features

| Feature | Description |
|:--------|:------------|
| **🤖 RL Gatekeeper** | A PPO (Proximal Policy Optimization) agent that audits every prediction and decides to `APPROVE`, `DENY`, or `ESCALATE` based on real-time fairness metrics like Demographic Parity Ratio |
| **🔍 Explainability (XAI)** | SHAP-powered explanations for every intervention, generating waterfall plots and natural language reasoning for regulatory transparency |
| **📊 Live Dashboard** | Real-time Accuracy vs. Fairness charts with glassmorphism design for compliance officers to monitor the system |
| **📜 Audit Log** | Immutable record of all decisions, interventions, and explanations for EU AI Act compliance |
| **🌊 Bias Drift Simulation** | Inject synthetic bias to demonstrate the self-healing capabilities in a controlled demo environment |
| **⚖️ Dynamic Fairness Optimization** | Composite reward function that optimizes for both accuracy and fairness simultaneously |

### Technical Implementation
- **Base Model:** XGBoost classifier (intentionally biased for demonstration)
- **RL Agent:** Stable-Baselines3 PPO with custom fairness environment
- **XAI Engine:** SHAP TreeExplainer for feature attribution
- **Backend:** FastAPI with real-time simulation endpoints
- **Frontend:** Next.js + Recharts dashboard with dark mode

---

## 3. Reinforcement Learning Layer (Deep Dive)

### 🧠 The Universal RL Gatekeeper

The core innovation of FairFlow is its **dataset-agnostic RL agent**. Unlike traditional ML fairness approaches that are tightly coupled to specific datasets and features, our RL agent uses **statistical fairness summaries** instead of raw features, making it **transferable across any domain** without retraining.

### 📊 State Space (12-Dimensional Universal State)

The RL agent observes a **fixed 12-dimensional state vector** that captures the essential fairness dynamics regardless of the underlying dataset:

| Dimension | Name | Description | Range |
|:---------:|:-----|:------------|:------|
| 0 | `base_prediction` | Base model's binary prediction | 0 or 1 |
| 1 | `base_confidence` | Base model's confidence score | [0, 1] |
| 2 | `protected_value` | Protected attribute of current sample | 0=Unprivileged, 1=Privileged |
| 3 | `current_dpr` | Rolling Demographic Parity Ratio (normalized) | [0, 1] |
| 4 | `current_tpr_diff` | Rolling True Positive Rate difference between groups | [0, 1] |
| 5 | `current_fpr_diff` | Rolling False Positive Rate difference between groups | [0, 1] |
| 6 | `privileged_approval` | Approval rate for privileged group (rolling) | [0, 1] |
| 7 | `unprivileged_approval` | Approval rate for unprivileged group (rolling) | [0, 1] |
| 8 | `intervention_rate` | Recent rate of RL agent interventions | [0, 1] |
| 9 | `group_ratio` | Proportion of unprivileged samples in data | [0, 1] |
| 10 | `consecutive_same` | Consecutive same-group predictions (normalized) | [0, 1] |
| 11 | `confidence_gap` | Model confidence gap between groups | [0, 1] |

> **Key Insight:** None of these 12 dimensions depend on the specific feature columns of the dataset. This means an agent trained on loan data can be deployed on hiring data, insurance claims, or any other binary classification task!

### 🎮 Action Space

The agent has 3 discrete actions:

| Action | Name | Effect |
|:------:|:-----|:-------|
| 0 | **APPROVE** | Accept base model's prediction as-is |
| 1 | **DENY** | Override to rejection (force output = 0) |
| 2 | **ACCEPT** | Override to approval (force output = 1) |

### ⚖️ Composite Reward Function

The agent is trained with a carefully designed reward function that balances accuracy and fairness:

```
R = w_acc × Accuracy_Reward + w_fair × Fairness_Reward + Intervention_Penalty
```

Where:
- **Accuracy Reward:** `+1` for correct prediction, `-1` for incorrect
- **Fairness Reward:** 
  - `+0.5` bonus when DPR ≥ 0.8 (legal threshold)
  - `-2.0 × (threshold - DPR)` penalty when below threshold
  - `+0.5` bonus for approving unprivileged applicants when DPR is low
- **Intervention Penalty:** `-0.1` small penalty to prefer minimal intervention
- **Default Weights:** `w_acc = 0.4`, `w_fair = 0.5`

### 🌐 Dataset-Agnostic Design

**Why is it dataset-agnostic?**

| Traditional Approach | FairFlow Approach |
|:---------------------|:------------------|
| State = raw features (age, income, etc.) | State = statistical summaries only |
| Tied to specific feature columns | Works with any tabular dataset |
| Requires retraining for new domains | Zero-shot transfer across domains |
| State dimension varies per dataset | Fixed 12-dimensional state always |

**How it works:**
1. The base model (XGBoost, neural network, etc.) processes raw features and produces a prediction
2. FairFlow extracts only the **prediction**, **confidence**, **protected attribute**, and **fairness metrics**
3. The RL agent makes decisions based on these universal signals
4. No raw features ever enter the RL agent's state

### 🔑 Key Features of the RL Layer

| Feature | Description |
|:--------|:------------|
| **Rolling Window Metrics** | DPR, TPR, FPR calculated over last 50 decisions for real-time tracking |
| **PPO Algorithm** | Proximal Policy Optimization from Stable-Baselines3 for stable training |
| **Configurable Fairness Profiles** | Adjust `accuracy_weight` and `fairness_weight` for different use cases |
| **Fairness Threshold** | Default 0.8 DPR (80% rule from US Equal Employment Opportunity Commission) |
| **Synthetic Data Training** | Can train on synthetic biased data before deploying on real data |
| **Human Escalation** | ESCALATE action for borderline cases requiring human review |

### 📈 Training Pipeline

```
1. Generate/Load biased base model predictions
           ↓
2. Create UniversalFairnessEnv with predictions + protected attributes
           ↓
3. Train PPO agent with composite reward
           ↓
4. Evaluate on held-out test set
           ↓
5. Deploy as wrapper around any base model
```

---

## 4. Pipeline Execution Diagram Description

### Diagram File: `pipeline_diagram_description.md`
*See the detailed description below for generating the pipeline diagram*

---

## 5. Architecture Diagram Description

### Diagram File: `architecture_diagram_description.md`
*See the detailed description below for generating the architecture diagram*

---

## 📎 Additional Notes for Presentation

### Demo Scenario (60-Second Story Arc)

1. **Baseline (Chaos):** Run the biased base model. Dashboard shows Accuracy is high (~85%), but Fairness (DPR) crashes below legal threshold (0.8)
2. **Activate FairFlow:** Toggle the switch on the dashboard
3. **Recovery:** Watch RL agent restore fairness while maintaining acceptable accuracy (~82%)
4. **Explainability:** Click any intervention to see SHAP explanation

### Target Track Alignment

**Track 2: Safe, Trusted & Responsible Technology**

| Hackathon Theme | FairFlow Feature |
|:----------------|:-----------------|
| Bias detection and fairness in decision-making | Core RL agent monitors Demographic Parity, Equalized Odds |
| Privacy-preserving and secure data management | Model-agnostic wrapper; does not access raw training data |
| Transparent and explainable decision-support | SHAP-based explanations for every RL intervention |
| Responsible technology governance and compliance | Immutable Audit Log, designed for EU AI Act Article 9 |

---

## 🛠️ Tech Stack Summary

| Layer | Technology |
|:------|:-----------|
| ML Model | XGBoost |
| RL Agent | Stable-Baselines3 (PPO) |
| RL Environment | OpenAI Gymnasium |
| XAI | SHAP |
| Backend | FastAPI |
| Frontend | Next.js + Recharts |
| Database | SQLite |
