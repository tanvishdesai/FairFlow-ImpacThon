# FairFlow: The RL-Driven Adaptive Bias Firewall

## 📖 Vision Statement

**FairFlow** is an enterprise-grade AI Governance Platform that acts as a "Self-Healing Bias Firewall" for black-box machine learning models. It uses Deep Reinforcement Learning (RL) to dynamically balance predictive accuracy with regulatory fairness requirements, ensuring continuous compliance with regulations like the **EU AI Act** and **GDPR**.

The core idea is a **"Gatekeeper Agent"** that sits between a deployed model and the end-user. It audits predictions in real-time and can intervene (accept, override, or escalate to human review) to prevent biased outcomes without requiring a full model retrain.

---

## 🎯 Target Track & Alignment

**Primary Track:** Track 2: Safe, Trusted & Responsible Technology

| Hackathon Theme                                  | FairFlow Feature                                           |
| :----------------------------------------------- | :--------------------------------------------------------- |
| Bias detection and fairness in decision-making  | Core RL agent monitors Demographic Parity, Equalized Odds. |
| Privacy-preserving and secure data management   | Model-agnostic wrapper; does not access raw training data. |
| Transparent and explainable decision-support    | SHAP-based explanations for every RL intervention.         |
| Responsible technology governance and compliance | Immutable Audit Log, designed for EU AI Act Article 9.     |

---

## 🏗️ High-Level Architecture

```
┌─────────────────────┐      ┌───────────────────────────────────────────────────────┐
│                     │      │                 FairFlow Platform                     │
│  Corporate "Base"   │      │  ┌─────────────┐    ┌─────────────┐    ┌───────────┐  │
│   Model (XGBoost)   │─────▶│  │ RL Gatekeeper│───▶│ XAI Engine  │───▶│ Audit Log │  │
│   (Black Box)       │ API  │  │   (PPO/DQN)  │    │   (SHAP)    │    │  (Postgres)│  │
│                     │      │  └─────────────┘    └─────────────┘    └───────────┘  │
└─────────────────────┘      │        │ Decision: Accept / Override / Escalate       │
                             │        ▼                                              │
                             │  ┌───────────────────────────────────────────────┐    │
                             │  │              React Dashboard                   │    │
                             │  │  - Live Accuracy vs. Fairness Chart            │    │
                             │  │  - Intervention History & SHAP Explanations   │    │
                             │  │  - Human Review Queue                          │    │
                             │  └───────────────────────────────────────────────┘    │
                             └───────────────────────────────────────────────────────┘
```

### Component Breakdown

1.  **Bias Simulator/Base Model:** A pre-trained XGBoost model on the "German Credit Data" dataset, intentionally left without fairness constraints to act as the "problem" we are solving.
2.  **RL Environment (OpenAI Gym):** A custom environment where the agent's state is the model's prediction + applicant features, and the action is `APPROVE`, `DENY`, or `ESCALATE`.
3.  **RL Gatekeeper (Ray RLLib / Stable-Baselines3):** A PPO or DQN agent trained with a composite reward: `R = w_acc * Accuracy_Reward + w_fair * Fairness_Penalty`.
4.  **XAI Engine (SHAP):** Generates feature attribution plots for every prediction, especially when an intervention occurs.
5.  **Backend (FastAPI):** Exposes `/predict`, `/audit`, and `/metrics` endpoints.
6.  **Frontend (Next.js + Recharts):** A real-time dashboard for the Chief Risk Officer persona.
7.  **Database (PostgreSQL):** Stores the immutable audit log for compliance.

---

## 🛠️ Tech Stack

| Layer          | Technology                             | Justification                                                     |
| :------------- | :------------------------------------- | :---------------------------------------------------------------- |
| **ML Model**   | XGBoost / LightGBM                     | Industry standard for tabular data (loans, insurance, HR).        |
| **RL Agent**   | Stable-Baselines3 (PPO) or Ray RLLib   | Modern, well-documented RL libraries with GPU support.            |
| **RL Env**     | OpenAI Gymnasium                       | Standard for custom simulation environments.                      |
| **XAI**        | SHAP                                   | Academically backed, visually intuitive explanations.             |
| **Backend**    | FastAPI (Python)                       | High-performance async API, great for ML serving.                 |
| **Frontend**   | Next.js (React) + Recharts / Chart.js | SSR for performance, powerful charting for real-time data.        |
| **Database**   | SQLite (Dev) / PostgreSQL (Prod)       | Relational DB ideal for structured audit logs.                    |
| **Task Queue** | Celery + Redis (Optional)              | For async processing of SHAP explanations if needed.              |

---

## ⚖️ Core Fairness Metrics

The RL agent will be trained to optimize for the following metrics, which will be displayed on the dashboard:

1.  **Demographic Parity Ratio (DPR):** `P(Approve | Group A) / P(Approve | Group B)`. Target: `0.8 < DPR < 1.25`.
2.  **Equalized Odds:** Difference in True Positive Rates (TPR) and False Positive Rates (FPR) across groups.
3.  **AUC-ROC / Accuracy:** The primary performance metric of the base model, which we want to preserve.

---

## 🚀 Demo Scenario ("The Story Arc")

This 60-second demo is the core of the presentation:

1.  **Baseline (Chaos):** Run the biased base model on a stream of test data. The dashboard shows Accuracy is high (Green line stable ~85%), but Fairness (Blue line) crashes below the legal threshold (0.8 DPR).
2.  **Activate FairFlow:** Click the "Activate FairFlow" toggle.
3.  **Recovery:** Watch in real-time as the RL Gatekeeper starts intervening. The Blue (Fairness) line rises and stabilizes above 0.8. The Green (Accuracy) line dips slightly but remains acceptable (~82%). The system has found the optimal trade-off.
4.  **Explainability Drill-Down:** Click on a specific "Intervention" event. The modal shows the applicant's data, the base model's "DENY" decision, FairFlow's "APPROVE" override, and the SHAP waterfall chart explaining that "Zip Code" (a proxy for a protected attribute) was the main factor.

---

## 📁 Project Directory Structure (Proposed)

```
fairflow/
├── PLANNING.md          # <-- You are here
├── TASK.md
├── README.md
│
├── data/
│   ├── raw/             # Original dataset (german_credit.csv)
│   └── processed/       # Train/Val/Test splits
│
├── models/
│   ├── base_model/      # Trained XGBoost model (biased)
│   └── rl_agent/        # Trained RL gatekeeper (PPO checkpoints)
│
├── src/
│   ├── environment/     # Custom OpenAI Gym environment
│   │   └── fairness_env.py
│   ├── agents/          # RL agent training scripts
│   │   └── train_ppo.py
│   ├── explainability/  # SHAP integration
│   │   └── shap_explainer.py
│   └── utils/           # Fairness metric calculators, data loaders
│       └── metrics.py
│
├── backend/
│   ├── main.py          # FastAPI app
│   ├── routes/
│   └── services/
│
└── frontend/
    ├── app/             # Next.js pages
    └── components/      # React components (Charts, etc.)
```

---

## ⚠️ Constraints & Risks

| Risk                           | Mitigation                                                                 |
| :----------------------------- | :------------------------------------------------------------------------- |
| RL training instability        | Use pre-trained agent; fallback to simple rule-based thresholds for demo. |
| SHAP is slow for large models | Pre-compute explanations or use `fast=True` mode. Cache results.          |
| Demo data drift simulation    | Prepare a fixed "drift injection" script that works deterministically.    |
| Time constraint (10 hours)    | Prioritize Core RL Loop > Dashboard > Polish.                             |

---

## 📚 Key References

-   **Fairness Metrics:** Aequitas (Fair ML Toolkit by UChicago)
-   **EU AI Act:** Article 9 (Risk Management System)
-   **RL for Fairness:** "Fairness in Machine Learning with Tractable Models" (academic paper)
-   **Dataset:** [German Credit Data (UCI)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
