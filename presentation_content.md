# FairFlow: The RL-Driven Adaptive Bias Firewall
## ImpactThon Presentation Content

---

## Slide 1: Problem Statement

**Track Number:** 2

**Track Title:** Safe, Trusted & Responsible Technology

**Video Link:** *(Optional)*

**Team ID:** *(Fill in)*

**Team Name:** *(Fill in)*

**Institute Name:** *(Fill in)*

---

## Slide 2: Team Details

**1. Member 1 Name**
- *Team Leader*
- Enrollment Number: *(Fill in)*
- Department, Institute Name, KSV

**2. Member 2 Name**
- *Co-Team Leader*
- Enrollment Number: *(Fill in)*
- Department, Institute Name, KSV

**3. Member 3 Name**
- *Team Member*
- Enrollment Number: *(Fill in)*
- Department, Institute Name, KSV

**4. Member 4 Name**
- *Team Member*
- Enrollment Number: *(Fill in)*
- Department, Institute Name, KSV

**5. Guide Name**
- *Guide*
- Department, Institute Name, KSV

---

## Slide 3: Idea Details – Proposed Solution

- **FairFlow** is a "Self-Healing Bias Firewall" that sits between deployed AI models and end-users, ensuring continuous fairness compliance in real-time
- Uses **Deep Reinforcement Learning (PPO)** to dynamically adjust decision thresholds, maintaining an optimal balance between Accuracy (Profit) and Fairness (Compliance)
- **Gatekeeper Agent** audits each prediction and decides to `APPROVE`, `DENY`, or `ESCALATE` based on real-time fairness metrics like Demographic Parity and Equalized Odds
- Every RL intervention is logged with **SHAP (Shapley Additive Explanations)** for transparent, explainable decision-making
- **Unique Innovation:** Unlike static bias-fixing methods, FairFlow continuously adapts to data drift in production, automatically correcting bias without retraining the base model

---

## Slide 4: Technical Approach – Technologies & Methodology

### Technologies Used:

| Layer | Technology |
|-------|------------|
| Base ML Model | XGBoost (deliberately biased for demo) |
| RL Agent | Stable-Baselines3 (PPO Algorithm) |
| RL Environment | OpenAI Gymnasium (Custom Fairness Env) |
| Explainability (XAI) | SHAP |
| Backend | FastAPI (Python) |
| Frontend | Next.js + Recharts |
| Database | SQLite |

### Step-by-Step Methodology:

1. **Data Preparation** – Load Adult Census Income Dataset, preprocess and split into training/validation/test sets
2. **Bias Simulation** – Train XGBoost classifier without fairness constraints to create "biased" base model
3. **RL Environment Design** – Build custom Gym environment where state = prediction + features, action = approve/deny/escalate
4. **RL Agent Training** – Train PPO agent with composite reward: `R = w_acc × Accuracy + w_fair × Fairness_Penalty`
5. **XAI Integration** – Generate SHAP explanations for every intervention
6. **Backend Development** – FastAPI endpoints for `/predict`, `/audit`, `/metrics`
7. **Dashboard Creation** – Real-time React dashboard with live Accuracy vs. Fairness charts

---

## Slide 5: Architecture – Proposed Architecture

```
┌─────────────────────────┐      ┌──────────────────────────────────────────────────────────┐
│                         │      │                    FairFlow Platform                      │
│   Corporate "Base"      │      │  ┌─────────────────┐  ┌─────────────┐  ┌───────────────┐  │
│   Model (XGBoost)       │─────▶│  │  RL Gatekeeper  │──│ XAI Engine  │──│  Audit Log    │  │
│   (Black Box)           │ API  │  │   (PPO Agent)   │  │   (SHAP)    │  │   (SQLite)    │  │
│                         │      │  └────────┬────────┘  └─────────────┘  └───────────────┘  │
└─────────────────────────┘      │           │ Decision: Accept / Override / Escalate        │
                                 │           ▼                                                │
                                 │  ┌────────────────────────────────────────────────────┐   │
                                 │  │              React Dashboard (Next.js)             │   │
                                 │  │  • Live Accuracy vs. Fairness Charts (Recharts)    │   │
                                 │  │  • Intervention History & SHAP Explanations        │   │
                                 │  │  • Real-time Bias Monitoring                       │   │
                                 │  └────────────────────────────────────────────────────┘   │
                                 └──────────────────────────────────────────────────────────┘
```

### Data Flow:
1. **Input** → User data sent to Base Model
2. **Prediction** → XGBoost generates initial decision
3. **Audit** → RL Gatekeeper evaluates fairness impact
4. **Intervention** → Agent approves, overrides, or escalates
5. **Explain** → SHAP generates feature attribution
6. **Log** → All decisions stored in Audit Log
7. **Monitor** → Dashboard displays real-time metrics

---

## Slide 6: Feasibility and Viability – Feasibility & Challenges

### Feasibility Analysis:
- **Technical Feasibility:** Built using well-established frameworks (Stable-Baselines3, XGBoost, FastAPI, Next.js) with proven stability
- **Data Feasibility:** Uses publicly available Adult Census Income Dataset (UCI Repository) for reproducible demonstrations
- **Resource Feasibility:** Runs on standard hardware; no GPU required for inference

### Potential Challenges & Mitigation Strategies:

| Challenge | Risk Level | Mitigation Strategy |
|-----------|------------|---------------------|
| RL Training Instability | Medium | Pre-trained agent checkpoint; fallback to rule-based thresholds |
| SHAP Computation Latency | Low | Use `fast=True` mode; cache frequent explanations |
| Real-time Performance | Medium | Async processing with FastAPI; optimized inference pipeline |
| Data Drift Handling | Low | Continuous monitoring; periodic agent retraining capability |
| Demo Reliability | Low | Deterministic "bias injection" scripts for controlled demonstrations |

---

## Slide 7: Impact and Benefits – Impact & Benefits

### Expected Impact on Target Audience:

- **Banks & Financial Institutions:** Ensures loan/credit decisions comply with EU AI Act and GDPR anti-discrimination requirements
- **HR & Recruitment Firms:** Prevents hiring algorithms from discriminating based on protected attributes (gender, age, race)
- **Insurance Companies:** Guarantees fair premium/claim decisions across demographic groups
- **Compliance Officers:** Provides real-time dashboard and audit trail for regulatory reporting

### Key Benefits:

| Benefit Type | Description |
|--------------|-------------|
| **Regulatory Compliance** | Automatic alignment with EU AI Act Article 9, GDPR fairness mandates |
| **Economic** | Reduces legal risk of discrimination lawsuits; avoids costly model retraining |
| **Social** | Promotes equitable AI decisions across all demographic groups |
| **Operational** | Self-healing capability reduces manual intervention; continuous monitoring |
| **Transparency** | SHAP-based explanations provide clear audit trail for every decision |

---

## Slide 8: Comparison with Existing System – Comparison

| Feature | Traditional Bias Mitigation | FairFlow (Our Solution) |
|---------|----------------------------|-------------------------|
| **Approach** | Static, one-time fix at training | Dynamic, real-time adaptation |
| **Data Drift Handling** | Requires complete model retrain | Auto-corrects using RL Gatekeeper |
| **Explainability** | Limited or none | Full SHAP explanations per decision |
| **Audit Trail** | Manual logging | Automatic, immutable audit log |
| **Integration** | Requires access to model internals | Model-agnostic wrapper (Black-box compatible) |
| **Compliance** | Periodic manual audits | Continuous real-time compliance monitoring |
| **Human-in-Loop** | Not supported | "Escalate" action for uncertain cases |
| **Performance Trade-off** | Fixed at training time | Dynamically optimized Accuracy-Fairness balance |
| **Deployment** | Replace entire model | Plug-and-play middleware layer |
| **Cost** | High retraining costs | Low operational overhead |

---

*Note: Delete this content guidance before submitting. Save final PPT as PDF only.*
