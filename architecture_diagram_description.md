# FairFlow Architecture Diagram - Image Generation Prompt

## 📌 Purpose
This document describes the visual architecture diagram showing FairFlow's technical components and their interactions. Use this description with an AI image generation model to create the diagram.

---

## 🎨 Visual Style & Design Guidelines


### Layout Structure
- **Orientation:** Vertical layers with horizontal spread within layers (portrait or landscape)
- **Dimensions:** 16:9 or 4:3 aspect ratio for presentation compatibility
- **Layer Organization:** Top-to-bottom representing the stack:
  1. **Top Layer:** User Interface / Frontend
  2. **Middle Layer:** API Gateway & Business Logic
  3. **Core Layer:** ML/RL Engine (the brain)
  4. **Bottom Layer:** Data & Storage

---

## 📐 Structure & Layout Description

### Layer 1: USER INTERFACE LAYER (Top - 20% height)

**Visual:** Wide banner-like section at the top

#### 1.1 Compliance Dashboard (Primary UI)
- **Position:** Center-left
- **Visual:** Browser window mockup with dark theme
- **Icon:** Monitor with chart icon
- **Label:** "Compliance Dashboard"
- **Sub-label:** "Next.js + Recharts"
- **Inside Preview (mini-mockup):**
  - Two line charts icon (Accuracy vs Fairness)
  - 4 small metric card icons
  - Toggle switch icon
  - Table/log icon
- **Badge:** React logo

#### 1.2 Human Review Interface (Secondary UI)
- **Position:** Center-right
- **Visual:** Smaller browser window mockup
- **Icon:** Person with clipboard
- **Label:** "Human Review Queue"
- **Sub-label:** "Escalated Cases"
- **Visual Elements:** Queue list with case cards

#### 1.3 External Consumer Application
- **Position:** Far right
- **Visual:** Mobile phone or generic app icon
- **Icon:** Smartphone with user icon
- **Label:** "Consumer App"
- **Sub-label:** "3rd Party Integration"

**Connecting Elements:**
- All three UI elements have downward arrows to the API Layer
- Labels on arrows: "WebSocket", "REST API", "API Calls"

---

### Layer 2: API GATEWAY LAYER (Upper-Middle - 15% height)

**Visual:** Horizontal band with hexagonal or rounded rectangle shapes

#### 2.1 FastAPI Backend (Central Gateway)
- **Position:** Center, spanning wide
- **Visual:** Large rounded rectangle with FastAPI logo/icon
- **Icon:** Lightning bolt or rocket (speed)
- **Label:** "FastAPI Backend"
- **Sub-label:** "REST API Gateway"
- **Color:** Blue gradient border

**API Endpoints (inside or as sub-nodes):**

| Endpoint | Icon | Color |
|:---------|:-----|:------|
| `/api/predict` | Arrow-right | Green |
| `/api/metrics` | Chart-bar | Blue |
| `/api/audit-log` | List | Amber |
| `/api/explain/{id}` | Lightbulb | Purple |
| `/api/fairflow/toggle` | Toggle | Cyan |
| `/api/simulate/*` | Play button | Gray |

**Show as:** Small pills or tags inside or below the FastAPI box

#### 2.2 CORS Middleware
- **Position:** Small badge on FastAPI box
- **Icon:** Lock or shield
- **Label:** "CORS"
- **Visual:** Subtle badge/tag

---

### Layer 3: CORE ML/RL ENGINE LAYER (Center - 40% height) ⭐ HERO SECTION

**Visual:** The largest, most visually prominent section with three major interconnected components

**Background:** Subtle gradient or slightly different shade to emphasize importance

#### 3.1 Base Model (Left)
- **Position:** Left side of core layer
- **Visual:** Rectangular box with tree/gradient icon
- **Icon:** Decision tree or XGBoost logo-style icon
- **Label:** "Base Model"
- **Sub-label:** "XGBoost Classifier"
- **Color Accent:** Blue
- **Details Inside:**
  - "models/base_model/" path
  - "Loan Default Prediction"
  - Warning icon: "Potentially Biased"
- **Input Arrow:** From API layer (labeled "Features")
- **Output Arrow:** To RL Gatekeeper (labeled "Raw Prediction P(default)")

#### 3.2 RL Gatekeeper (Center - HERO COMPONENT) ⭐
- **Position:** Dead center, largest component
- **Visual:** Prominent box with glowing cyan border, distinctive shape (hexagon or rounded with badge)
- **Icon:** Shield with brain, robot head, or AI agent icon
- **Label:** "RL Gatekeeper"
- **Sub-label:** "PPO Agent (Stable-Baselines3)"
- **Color Accent:** Vibrant cyan with glow effect
- **Glow:** Soft outer glow to draw attention

**Internal Structure (show inside the box):**

```
┌─────────────────────────────────────┐
│         RL GATEKEEPER (PPO)          │
├─────────────────────────────────────┤
│  ┌───────────┐    ┌───────────────┐ │
│  │   State   │    │    Policy     │ │
│  │ Encoder   │───▶│   Network     │ │
│  └───────────┘    └───────────────┘ │
│       │                   │         │
│       ▼                   ▼         │
│  ┌────────────────────────────┐     │
│  │        Action Space        │     │
│  │  ✅ APPROVE  ❌ DENY  🔄 ESC │     │
│  └────────────────────────────┘     │
├─────────────────────────────────────┤
│  Observation: [Prediction, Features,│
│                DPR, Fairness Metrics]│
└─────────────────────────────────────┘
```

**Inputs (arrows coming in):**
- From Base Model: "Raw Prediction"
- From Fairness Monitor (below): "Current DPR"
- From API: "Applicant Features"

**Outputs (arrows going out):**
- To XAI Engine: "Decision + Context"
- To Audit Log: "Action Taken"

#### 3.3 XAI Engine (Right)
- **Position:** Right side of core layer
- **Visual:** Box with lightbulb or explanation icon
- **Icon:** Magnifying glass with SHAP waterfall mini-icon
- **Label:** "XAI Engine"
- **Sub-label:** "SHAP Explainer"
- **Color Accent:** Purple

**Details Inside:**
- "TreeExplainer"
- "Feature Attribution"
- Mini waterfall chart preview
- "Natural Language Explanation"

**Outputs:**
- To Audit Log: "Explanation Data"
- To API/Dashboard: "SHAP Plot (base64)"

#### 3.4 Connecting Lines in Core Layer
- Base Model → RL Gatekeeper: Thick line, labeled "Prediction"
- RL Gatekeeper → XAI Engine: Thick line, labeled "Decision Context"
- All three connect down to the data layer

---

### Layer 4: FAIRNESS MONITORING LAYER (Lower-Middle - 15% height)

**Visual:** Horizontal band with gauge/monitoring icons

#### 4.1 Fairness Metrics Calculator
- **Position:** Center-left
- **Visual:** Box with formula/calculation icon
- **Icon:** Scale/balance icon
- **Label:** "Fairness Metrics"
- **Sub-label:** "src/utils/metrics.py"
- **Details:**
  - Demographic Parity Ratio (DPR)
  - Equalized Odds
  - Accuracy Score
- **Visual:** Mini gauge showing DPR in green zone (0.8-1.25)

#### 4.2 Rolling Window Tracker
- **Position:** Center
- **Visual:** Smaller box with window/buffer icon
- **Icon:** Sliding window or history icon
- **Label:** "Rolling Window"
- **Sub-label:** "Last N decisions"
- **Arrow:** Feeds into RL Gatekeeper's state

#### 4.3 Bias Drift Detector (Optional)
- **Position:** Center-right
- **Visual:** Small box with alert/warning capability
- **Icon:** Warning triangle with trend line
- **Label:** "Drift Monitor"
- **Sub-label:** "Bias Simulation"
- **Note:** Red accent when drift detected

---

### Layer 5: DATA & STORAGE LAYER (Bottom - 10% height)

**Visual:** Foundation layer with database and storage icons

#### 5.1 SQLite Database
- **Position:** Center-left
- **Visual:** Cylinder database icon
- **Icon:** Database with lock
- **Label:** "SQLite Database"
- **Sub-label:** "Audit Log Storage"
- **Color Accent:** Amber/Orange
- **Details:**
  - "predictions" table icon
  - "interventions" table icon
  - Timestamp icon
- **Badge:** "Immutable Records"

#### 5.2 Model Storage
- **Position:** Center
- **Visual:** Folder or file storage icon
- **Icon:** Folder with model file icon
- **Label:** "Model Storage"
- **Sub-label:** "models/"
- **Details:**
  - "base_model.pkl"
  - "rl_agent.zip"
  - "shap_explainer.pkl"

#### 5.3 Dataset Storage
- **Position:** Center-right
- **Visual:** Data file icon
- **Icon:** CSV/table icon
- **Label:** "Dataset"
- **Sub-label:** "data/processed/"
- **Details:**
  - "train.csv"
  - "test.csv"
  - UCI Adult Census reference

---

## 🔗 Connection Lines & Arrows

### Primary Data Flow (Thick Lines)
1. Consumer App → FastAPI: "API Request"
2. FastAPI → Base Model: "Features"
3. Base Model → RL Gatekeeper: "Prediction"
4. RL Gatekeeper → XAI Engine: "Decision + Context"
5. XAI Engine → FastAPI: "Explanation"
6. FastAPI → Dashboard: "Response"

### Secondary Flows (Medium Lines)
- RL Gatekeeper ↔ Fairness Metrics: Bidirectional "State Update"
- All Core Components → Audit Log: "Log Events"
- Dashboard → FastAPI: "Toggle FairFlow"

### Feedback Loops (Dashed Lines)
- Rolling Window → RL Gatekeeper: "Current DPR"
- Drift Monitor → Dashboard: "Alert"

### Arrow Styling
| Type | Style | Color | Use |
|:-----|:------|:------|:----|
| Data flow | Solid, thick | White/Light | Primary paths |
| Control | Solid, medium | Cyan | Commands/toggles |
| Logging | Dashed, thin | Amber | Audit trails |
| Feedback | Dotted | Purple | Metrics loops |

---

## 📝 Labels & Annotations

### Layer Headers (Large, Bold, Left-aligned)
- "🖥️ PRESENTATION LAYER"
- "🔌 API LAYER"
- "🧠 AI/ML ENGINE"
- "📊 MONITORING"
- "💾 DATA LAYER"

### Technology Badges (Small pills/tags)
- Show tech stack logos or names as small badges:
  - Next.js, React, Recharts (Frontend)
  - FastAPI, Python (Backend)
  - XGBoost, Stable-Baselines3, SHAP (ML)
  - SQLite, Pandas (Data)

### Compliance Badges
- EU AI Act icon near Audit Log
- "Article 9 Compliant" badge
- Shield icon for security/privacy

---

## 🖼️ Overall Composition Summary

Picture a modern, dark-themed enterprise architecture diagram with 5 clear horizontal layers:

1. **Top:** Sleek dashboard mockups and consumer app icons representing the user-facing layer

2. **Upper-Middle:** FastAPI gateway with endpoint pills showing the API surface

3. **Center (Hero):** The AI brain - three interconnected boxes with the RL Gatekeeper as the glowing centerpiece, flanked by Base Model (input) and XAI Engine (output)

4. **Lower-Middle:** Fairness monitoring with gauges and metrics calculators

5. **Bottom:** Database cylinders and storage icons as the foundation

**Visual Flow:** 
- Arrows flow generally top-to-bottom for requests
- Responses flow bottom-to-top
- Internal loops (fairness metrics) shown in the center
- Clear visual hierarchy with the RL Gatekeeper as the focal point

**Color Story:**
- Blue for ML/traditional components
- Cyan (glowing) for RL/innovation
- Purple for explainability
- Amber for data/storage
- Green for positive outcomes
- Dark background makes components pop

---

## 📏 Approximate Dimensions for Key Elements

| Element | Relative Size | Z-Order |
|:--------|:--------------|:--------|
| RL Gatekeeper | 20% width × 25% height | Front/Top (most prominent) |
| Base Model | 15% width × 20% height | Same level as XAI |
| XAI Engine | 15% width × 20% height | Same level as Base Model |
| FastAPI Box | 50% width × 8% height | Background |
| Database Icons | 10% each | Background/Foundation |
| Dashboard Mockup | 25% width × 15% height | Standard |

---

## 🎯 Key Visual Emphasis Points

1. **RL Gatekeeper Glow:** This should be the first thing the eye is drawn to - use a subtle but noticeable glow effect

2. **Flow Direction:** Make the primary prediction flow (Request → Predict → Decide → Explain → Respond) visually obvious

3. **Compliance Visual:** Include visible EU AI Act compliance indicators to emphasize regulatory readiness

4. **Technology Logos:** Subtle but visible logos/icons for XGBoost, SHAP, PPO to show technical sophistication

5. **Action Indicators:** The APPROVE/DENY/ESCALATE decision point should be clearly visible with traffic light colors (green/red/yellow)
