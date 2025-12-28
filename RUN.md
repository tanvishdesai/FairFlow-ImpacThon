# FairFlow - Run Scripts

## Prerequisites

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
cd ..
```

## Step 1: Train the Base Model

First, train the biased XGBoost model that FairFlow will correct:

```bash
python src/train_base_model.py
```

This will:
- Load the Adult Census dataset (32,561 samples)
- Train a biased classifier (DPR ≈ 0.32 - women approved at 32% the rate of men)
- Save to `models/base_model/xgboost_biased.joblib`

## Step 2: Train the RL Agent (Optional)

Train the PPO agent for bias mitigation:

```bash
python src/agents/train_ppo.py
```

This will:
- Create the custom Gym environment
- Train PPO for ~100k timesteps
- Save to `models/rl_agent/ppo_fairness_agent.zip`

**Note:** If you skip this step, the backend will use a rule-based fallback.

## Step 3: Start the Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

## Step 4: Start the Frontend

In a new terminal:

```bash
cd frontend
npm run dev
```

The dashboard will be available at http://localhost:3000

## Step 5: Run the Demo

1. Open http://localhost:3000 in your browser
2. Click **"Start Simulation"** to begin streaming test data
3. Watch the metrics update in real-time
4. Toggle **FairFlow OFF** to see the model become biased
5. Toggle **FairFlow ON** to see fairness recover
6. Click any row in the audit log to see SHAP explanations
7. Click **"Inject Drift"** to simulate a bias event

## Environment Variables

The frontend expects the backend at `http://localhost:8000`.
If you need to change this, create `frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://your-backend-url:8000
```
