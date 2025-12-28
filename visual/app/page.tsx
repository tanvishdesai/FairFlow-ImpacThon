"use client";

import { useState, useEffect, useRef } from "react";
import { 
  Activity, 
  Target, 
  Users, 
  AlertTriangle, 
} from "lucide-react";

import Header from "./components/Header";
import MetricsCard from "./components/MetricsCard";
import ControlPanel from "./components/ControlPanel";
import FairnessChart from "./components/FairnessChart";
import AuditLog from "./components/AuditLog";
import ExplanationModal from "./components/ExplanationModal";
import DriftAlert from "./components/DriftAlert";

import {
  useMetrics,
  useAuditLog,
  useSimulation,
  useFairFlow,
  useExplanation,
  useApiHealth,
} from "./hooks/useApi";

interface ChartDataPoint {
  time: string;
  accuracy: number;
  fairness: number;
}

export default function Dashboard() {
  // API Hooks
  const { isConnected } = useApiHealth();
  const { metrics } = useMetrics(1000);
  const { entries } = useAuditLog(50, 1000);
  const { status: simStatus, startSimulation, stopSimulation, injectDrift } = useSimulation(1000);
  const { status: fairflowStatus, toggleFairFlow } = useFairFlow();
  const { 
    explanation, 
    isLoading: explanationLoading, 
    fetchExplanation, 
    clearExplanation 
  } = useExplanation();

  // Chart data history
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [selectedPredictionId, setSelectedPredictionId] = useState<number | null>(null);
  
  // Drift state (client-side tracking)
  const [driftActive, setDriftActive] = useState(false);
  const [driftSamplesRemaining, setDriftSamplesRemaining] = useState(0);
  
  // Ref to track last prediction count to avoid duplicate updates
  const lastPredictionCount = useRef(0);

  // Update chart data when metrics change
  useEffect(() => {
    if (metrics && simStatus?.is_running) {
      // Only update if prediction count actually changed
      if (metrics.total_predictions !== lastPredictionCount.current) {
        lastPredictionCount.current = metrics.total_predictions;
        
        const timestamp = new Date().toLocaleTimeString('en-US', { 
          hour: '2-digit', 
          minute: '2-digit',
          second: '2-digit'
        });
        
        setChartData((prev) => {
          const newPoint = {
            time: timestamp,
            accuracy: metrics.accuracy,
            fairness: metrics.demographic_parity_ratio,
          };
          
          // Keep last 30 points for smooth chart
          const updated = [...prev, newPoint];
          return updated.slice(-30);
        });
      }
    }
  }, [metrics, simStatus?.is_running]);

  // Separate effect for drift countdown - triggered by prediction count changes
  useEffect(() => {
    if (driftActive && metrics) {
      const currentCount = metrics.total_predictions;
      if (currentCount !== lastPredictionCount.current) {
        setDriftSamplesRemaining(prev => {
          const newValue = Math.max(0, prev - 1);
          if (newValue <= 0) {
            setDriftActive(false);
          }
          return newValue;
        });
      }
    }
  }, [metrics?.total_predictions, driftActive]);

  // Handle explanation modal
  const handleSelectEntry = (id: number) => {
    setSelectedPredictionId(id);
    fetchExplanation(id);
  };

  const handleCloseModal = () => {
    setSelectedPredictionId(null);
    clearExplanation();
  };
  
  // Handle drift injection with visual feedback
  const handleInjectDrift = async () => {
    const duration = 50;
    await injectDrift(0.9, duration);
    setDriftActive(true);
    setDriftSamplesRemaining(duration);
  };

  // Computed values
  const fairflowActive = fairflowStatus?.active ?? true;
  const simulationRunning = simStatus?.is_running ?? false;
  const isFair = metrics?.demographic_parity_ratio 
    ? metrics.demographic_parity_ratio >= 0.8 
    : true;

  return (
    <div className="min-h-screen p-6">
      {/* Drift Alert Banner */}
      <DriftAlert 
        isActive={driftActive} 
        samplesRemaining={driftSamplesRemaining} 
      />
      
      <div className="max-w-[1800px] mx-auto">
        {/* Header */}
        <Header isConnected={isConnected} fairflowActive={fairflowActive} />

        {/* Main Grid */}
        <div className="grid grid-cols-12 gap-6">
          {/* Left Column - Metrics Cards */}
          <div className="col-span-12 lg:col-span-3 space-y-6">
            {/* Total Predictions */}
            <MetricsCard
              title="Total Predictions"
              value={metrics?.total_predictions ?? 0}
              icon={Activity}
              status="neutral"
            />

            {/* Accuracy */}
            <MetricsCard
              title="Model Accuracy"
              value={`${((metrics?.accuracy ?? 0) * 100).toFixed(1)}%`}
              subtitle="Ground truth match rate"
              icon={Target}
              trend={metrics?.accuracy && metrics.accuracy > 0.8 ? "up" : "neutral"}
              status={metrics?.accuracy && metrics.accuracy > 0.8 ? "good" : "warning"}
            />

            {/* Demographic Parity */}
            <MetricsCard
              title="Fairness (DPR)"
              value={(metrics?.demographic_parity_ratio ?? 1.0).toFixed(3)}
              subtitle={isFair ? "Above 0.8 threshold ✓" : "Below 0.8 threshold ⚠"}
              icon={Users}
              trend={isFair ? "up" : "down"}
              status={isFair ? "good" : "danger"}
            />

            {/* Interventions */}
            <MetricsCard
              title="Interventions"
              value={metrics?.total_interventions ?? 0}
              subtitle={`${((metrics?.intervention_rate ?? 0) * 100).toFixed(1)}% of decisions`}
              icon={AlertTriangle}
              status={metrics?.total_interventions && metrics.total_interventions > 0 ? "warning" : "neutral"}
            />

            {/* Approval Rates */}
            <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: '0.5s' }}>
              <h4 className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>Approval Rates by Group</h4>
              <div className="space-y-4">
                {/* Male */}
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span style={{ color: '#3b82f6' }}>Male (Privileged)</span>
                    <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                      {((metrics?.privileged_approval_rate ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div 
                      className="h-full rounded-full transition-all duration-500"
                      style={{ 
                        width: `${(metrics?.privileged_approval_rate ?? 0) * 100}%`,
                        backgroundColor: '#3b82f6'
                      }}
                    />
                  </div>
                </div>
                {/* Female */}
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span style={{ color: '#ec4899' }}>Female (Unprivileged)</span>
                    <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                      {((metrics?.unprivileged_approval_rate ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div 
                      className="h-full rounded-full transition-all duration-500"
                      style={{ 
                        width: `${(metrics?.unprivileged_approval_rate ?? 0) * 100}%`,
                        backgroundColor: '#ec4899'
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Center Column - Charts */}
          <div className="col-span-12 lg:col-span-6 space-y-6">
            {/* Fairness Chart */}
            <FairnessChart data={chartData} fairnessThreshold={0.8} />

            {/* Audit Log */}
            <AuditLog entries={entries} onSelectEntry={handleSelectEntry} />
          </div>

          {/* Right Column - Controls */}
          <div className="col-span-12 lg:col-span-3 space-y-6">
            {/* Control Panel */}
            <ControlPanel
              fairflowActive={fairflowActive}
              simulationRunning={simulationRunning}
              driftActive={driftActive}
              onToggleFairflow={() => toggleFairFlow(!fairflowActive)}
              onStartSimulation={() => startSimulation(2.0)}
              onStopSimulation={stopSimulation}
              onInjectDrift={handleInjectDrift}
            />

            {/* System Status */}
            <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: '0.6s' }}>
              <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>System Status</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span style={{ color: 'var(--text-secondary)' }}>Backend API</span>
                  <span className="flex items-center gap-2" style={{ color: isConnected ? '#10b981' : '#ef4444' }}>
                    <div className={`w-2 h-2 rounded-full`} style={{ backgroundColor: isConnected ? '#10b981' : '#ef4444' }} />
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span style={{ color: 'var(--text-secondary)' }}>RL Agent</span>
                  <span className="flex items-center gap-2" style={{ color: fairflowStatus?.rl_agent_loaded ? '#10b981' : '#f59e0b' }}>
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: fairflowStatus?.rl_agent_loaded ? '#10b981' : '#f59e0b' }} />
                    {fairflowStatus?.rl_agent_loaded ? 'Loaded' : 'Fallback'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span style={{ color: 'var(--text-secondary)' }}>Mode</span>
                  <span className="font-medium" style={{ color: '#6366f1' }}>
                    {fairflowStatus?.mode === 'rl_agent' ? 'PPO Agent' : 'Rule-Based'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span style={{ color: 'var(--text-secondary)' }}>Simulation</span>
                  <span className="flex items-center gap-2" style={{ color: simulationRunning ? '#10b981' : 'var(--text-secondary)' }}>
                    <div className={`w-2 h-2 rounded-full ${simulationRunning ? 'animate-pulse' : ''}`} style={{ backgroundColor: simulationRunning ? '#10b981' : 'var(--text-muted)' }} />
                    {simulationRunning ? 'Running' : 'Stopped'}
                  </span>
                </div>
                {driftActive && (
                  <div className="flex items-center justify-between">
                    <span style={{ color: '#ef4444' }}>⚠️ Bias Drift</span>
                    <span className="flex items-center gap-2 animate-pulse" style={{ color: '#ef4444' }}>
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#ef4444' }} />
                      Active ({driftSamplesRemaining} left)
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* How It Works */}
            <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: '0.7s' }}>
              <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>How FairFlow Works</h3>
              <div className="space-y-4 text-sm">
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold" style={{ backgroundColor: 'rgba(99, 102, 241, 0.2)', color: '#6366f1' }}>1</div>
                  <p style={{ color: 'var(--text-secondary)' }}>Base model makes prediction on applicant data</p>
                </div>
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold" style={{ backgroundColor: 'rgba(99, 102, 241, 0.2)', color: '#6366f1' }}>2</div>
                  <p style={{ color: 'var(--text-secondary)' }}>RL Gatekeeper evaluates current fairness metrics</p>
                </div>
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold" style={{ backgroundColor: 'rgba(99, 102, 241, 0.2)', color: '#6366f1' }}>3</div>
                  <p style={{ color: 'var(--text-secondary)' }}>Agent decides: Accept, Override to Approve, or Override to Deny</p>
                </div>
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold" style={{ backgroundColor: 'rgba(16, 185, 129, 0.2)', color: '#10b981' }}>✓</div>
                  <p style={{ color: 'var(--text-secondary)' }}>SHAP explains every intervention for compliance</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
          <p style={{ color: 'var(--text-secondary)' }}>FairFlow: The RL-Driven Adaptive Bias Firewall</p>
          <p className="text-xs mt-1">Built for EU AI Act & GDPR Compliance</p>
        </footer>
      </div>

      {/* Explanation Modal */}
      <ExplanationModal
        explanation={explanation}
        isOpen={selectedPredictionId !== null}
        onClose={handleCloseModal}
        isLoading={explanationLoading}
      />
    </div>
  );
}
