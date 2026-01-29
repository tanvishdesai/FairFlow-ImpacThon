# FairFlow Pipeline Execution Diagram - Image Generation Prompt

## 📌 Purpose
This document describes the visual diagram for FairFlow's complete pipeline from training the base model to deployment and consumer usage. Use this description with an AI image generation model to create the diagram.

---

## 🎨 Visual Style & Design Guidelines

## 📐 Structure & Layout Description

### Section 1: TRAINING PHASE (Left Side - "Offline")

**Position:** Left 25% of the diagram, with a subtle vertical dashed line separating it from the next section

**Components (Top to Bottom):**

#### 1.1 Data Preparation Block
- **Icon:** Database cylinder with data streams
- **Label:** "Adult Census Dataset"
- **Sub-label:** "Historical Financial Data"
- **Visual:** Small icons showing demographic attributes (gender, age symbols)
- **Arrow:** Flows down to Base Model Training

#### 1.2 Base Model Training Block
- **Icon:** Neural network or tree structure (representing XGBoost)
- **Label:** "Train Base Model"
- **Sub-label:** "XGBoost Classifier"
- **Visual:** Show it producing a model artifact (box with gears icon)
- **Output Arrow 1:** Flows right to "Deliberately Biased Model" (red-tinted box)
- **Output Arrow 2:** Flows down to RL Environment

#### 1.3 RL Environment & Agent Training Block
- **Icon:** Game controller or agent brain icon
- **Visual:** A loop/cycle showing:
  - "Custom Gym Environment" (box)
  - "State: [Prediction, Features, DPR]" (smaller box inside)
  - "Action: [APPROVE, DENY, ESCALATE]" (smaller box inside)
  - "Reward Function" (formula icon: R = w_acc × Accuracy + w_fair × Fairness)
- **Label:** "Train RL Gatekeeper"
- **Sub-label:** "PPO Agent (Stable-Baselines3)"
- **Output:** Arrow flowing to "Trained RL Agent" (cyan-glowing box)

---

### Section 2: INFERENCE/DEPLOYMENT PHASE (Center - "Real-Time")

**Position:** Center 50% of the diagram, this is the main focus with the most visual prominence

**Components (Main Horizontal Flow):**

#### 2.1 Consumer Request Entry
- **Icon:** Person silhouette or user avatar
- **Label:** "Consumer/Applicant"
- **Sub-label:** "Loan/Credit Application"
- **Visual:** Application form icon with personal data fields
- **Arrow:** Flowing right into the FairFlow Platform box

#### 2.2 The FairFlow Platform (Large Central Box - Hero Component)
- **Visual:** Large rounded rectangle with glassmorphism effect (semi-transparent with blur)
- **Border:** Gradient border (cyan to purple)
- **Header:** "FairFlow Platform" with logo/shield icon

**Inside the FairFlow Platform Box (sub-components left to right):**

##### 2.2.1 Base Model Inference
- **Icon:** Small XGBoost tree icon
- **Label:** "Black-Box Model"
- **Output:** Arrow labeled "Raw Prediction" → going to RL Gatekeeper
- **Note:** Small red warning indicator showing "May be biased"

##### 2.2.2 RL Gatekeeper (Central Hero Sub-Component)
- **Icon:** Shield with brain or robot face
- **Visual:** Glowing cyan border, most prominent element
- **Label:** "RL Gatekeeper (PPO)"
- **Inputs:** 
  - Raw prediction from Base Model
  - Current Fairness Metrics (DPR indicator)
  - Applicant Features
- **Processing Visual:** Decision tree showing 3 branches:
  - ✅ APPROVE (green)
  - ❌ DENY (red)  
  - 🔄 ESCALATE (yellow/orange)
- **Output:** Arrow to XAI Engine

##### 2.2.3 XAI Engine
- **Icon:** Magnifying glass with chart/explanation bubbles
- **Label:** "SHAP Explainer"
- **Visual:** Small waterfall chart preview icon
- **Output:** Arrow labeled "Explanation" flowing to both Audit Log and Final Decision

##### 2.2.4 Audit Log Database
- **Icon:** Database with lock/shield icon
- **Label:** "Immutable Audit Log"
- **Visual:** Stack of records with timestamp icons
- **Note:** Small EU flag icon with "EU AI Act Compliant"

#### 2.3 Final Decision Output
- **Position:** Right side of FairFlow Platform
- **Icon:** Checkmark in circle
- **Label:** "Fair Decision"
- **Visual:** Green glowing output
- **Arrow:** Flows to Consumer Response

#### 2.4 Consumer Response
- **Icon:** Person with notification/result
- **Label:** "Consumer Receives Decision"
- **Sub-elements:** 
  - Decision result (Approved/Denied)
  - Explanation link "Why this decision?"

---

### Section 3: MONITORING & FEEDBACK LOOP (Bottom Band)

**Position:** Bottom 20% of the diagram, spanning across

**Components:**

#### 3.1 Live Dashboard
- **Icon:** Monitor/screen with charts
- **Label:** "Compliance Dashboard"
- **Visual:** Mini preview showing:
  - Two line charts (Accuracy in green, Fairness/DPR in blue)
  - Metric cards (DPR: 0.95, Accuracy: 82%)
- **Arrow:** Connected to FairFlow Platform (bidirectional for control toggle)

#### 3.2 Fairness Metrics Monitor
- **Icon:** Gauge or dial
- **Label:** "Real-time Metrics"
- **Visual:** Circular gauge showing DPR value (green zone: 0.8-1.25)
- **Note:** "Demographic Parity Ratio"

#### 3.3 Human Review Queue (Optional Component)
- **Icon:** Person with clipboard
- **Label:** "Human Review"
- **Visual:** Queue of escalated cases
- **Arrow:** Receives ESCALATE decisions from RL Gatekeeper

#### 3.4 Feedback Arrow
- **Visual:** Large curved arrow from Dashboard back to RL Agent Training (showing continuous improvement)
- **Label:** "Model Refinement Loop"
- **Style:** Dashed line indicating optional/periodic process

---

## 🔗 Connection Lines & Arrows

### Arrow Styles
1. **Primary Flow Arrows:** Thick, solid, with gradient (white to cyan)
2. **Data Flow Arrows:** Medium thickness, solid, white
3. **Feedback Arrows:** Thin, dashed, purple/cyan
4. **Warning/Alert Connections:** Thin, dotted, red

### Arrow Labels (where applicable)
- "Raw Prediction"
- "Fair Decision"
- "SHAP Explanation"
- "Audit Record"
- "Metrics Update"

---

## 📝 Text Labels & Annotations

### Phase Labels (Large, Bold)
- "PHASE 1: TRAINING" (top-left)
- "PHASE 2: REAL-TIME INFERENCE" (center-top)
- "PHASE 3: MONITORING" (bottom-center)

### Timing Annotations
- "Offline (One-time)" near training section
- "Real-time (< 100ms)" near inference flow
- "Continuous" near monitoring

### Highlight Callouts
- Star/burst shape near RL Gatekeeper: "Core Innovation"
- Badge near XAI Engine: "Explainable AI"
- Shield icon near Audit Log: "Compliance Ready"

---

## 🖼️ Overall Composition Summary

Imagine a sleek, dark-themed infographic that looks like a premium fintech product diagram:

1. **Left Side (25%):** Shows the offline training setup with data flowing into both a base model (with a small red "biased" indicator) and an RL training loop that produces the smart gatekeeper agent

2. **Center (50%):** A large, glowing "FairFlow Platform" box containing the main inference pipeline - consumer request enters, passes through the base model, gets audited by the RL Gatekeeper (the visual hero), gets explained by SHAP, logged for compliance, and exits as a fair decision

3. **Right Side (25%):** The consumer receives their decision with an explanation, and the bottom shows a dashboard monitoring the whole system with a feedback loop for continuous improvement

4. **Visual Flow:** Clear left-to-right progression with numbered steps (1→2→3→4→5→6) showing the complete journey from model training to consumer receiving a fair, explained decision

---

## 📏 Approximate Dimensions for Key Elements

| Element | Relative Size | Position |
|:--------|:--------------|:---------|
| FairFlow Platform Box | 45% width × 50% height | Center |
| RL Gatekeeper | 15% width × 30% height | Inside platform, center-left |
| Training Section | 22% width | Left side |
| Consumer Elements | 15% width | Right side |
| Dashboard | 40% width × 15% height | Bottom center |
