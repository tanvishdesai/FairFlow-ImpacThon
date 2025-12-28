"use client";

import { Play, Square, AlertTriangle, Zap } from "lucide-react";

interface ControlPanelProps {
  fairflowActive: boolean;
  simulationRunning: boolean;
  driftActive?: boolean;
  onToggleFairflow: () => void;
  onStartSimulation: () => void;
  onStopSimulation: () => void;
  onInjectDrift: () => void;
}

export default function ControlPanel({
  fairflowActive,
  simulationRunning,
  driftActive = false,
  onToggleFairflow,
  onStartSimulation,
  onStopSimulation,
  onInjectDrift,
}: ControlPanelProps) {
  return (
    <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: '0.2s' }}>
      <h3 className="text-lg font-semibold mb-6 flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
        <Zap className="w-5 h-5 text-indigo-400" />
        Control Center
      </h3>
      
      {/* FairFlow Master Toggle */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-3">
          <div>
            <p className="font-medium" style={{ color: 'var(--text-primary)' }}>FairFlow Protection</p>
            <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>RL Gatekeeper bias mitigation</p>
          </div>
          <button
            onClick={onToggleFairflow}
            className={`toggle-switch ${fairflowActive ? 'active' : ''}`}
            aria-label="Toggle FairFlow"
          />
        </div>
        <div className={`p-3 rounded-lg ${
          fairflowActive 
            ? 'bg-green-500/10 border border-green-500/20' 
            : 'border'
        }`} style={{ borderColor: fairflowActive ? undefined : 'var(--border-subtle)' }}>
          <p className="text-sm" style={{ color: fairflowActive ? '#10b981' : 'var(--text-secondary)' }}>
            {fairflowActive 
              ? '✓ Active: RL agent monitoring all predictions for bias' 
              : '○ Inactive: Base model running without intervention'}
          </p>
        </div>
      </div>

      {/* Simulation Controls */}
      <div className="mb-6">
        <p className="font-medium mb-3" style={{ color: 'var(--text-primary)' }}>Simulation</p>
        <div className="flex gap-3">
          {!simulationRunning ? (
            <button onClick={onStartSimulation} className="btn-success flex-1">
              <Play className="w-5 h-5" />
              Start Simulation
            </button>
          ) : (
            <button onClick={onStopSimulation} className="btn-danger flex-1">
              <Square className="w-5 h-5" />
              Stop Simulation
            </button>
          )}
        </div>
        <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
          {simulationRunning 
            ? 'Processing test data samples...' 
            : 'Run predictions on Adult Census test data'}
        </p>
      </div>

      {/* Drift Injection */}
      <div>
        <p className="font-medium mb-3" style={{ color: 'var(--text-primary)' }}>Bias Drift Injection</p>
        <button 
          onClick={onInjectDrift} 
          disabled={!simulationRunning || driftActive}
          className={`w-full justify-center transition-all duration-300 ${
            driftActive 
              ? 'cursor-not-allowed opacity-75' 
              : 'cursor-pointer'
          }`}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            padding: '12px 24px',
            borderRadius: '12px',
            fontWeight: 600,
            border: driftActive ? '2px solid #ef4444' : '2px solid rgba(245, 158, 11, 0.3)',
            backgroundColor: driftActive ? 'rgba(239, 68, 68, 0.2)' : 'transparent',
            color: driftActive ? '#ef4444' : '#f59e0b',
            boxShadow: driftActive ? '0 0 20px rgba(239, 68, 68, 0.3)' : 'none',
            animation: driftActive ? 'pulse 2s infinite' : 'none',
          }}
        >
          <AlertTriangle className={`w-5 h-5 ${driftActive ? 'animate-bounce' : ''}`} />
          {driftActive ? '⚠️ DRIFT ACTIVE' : 'Inject Bias Drift'}
        </button>
        <p className="text-xs mt-2" style={{ color: driftActive ? '#ef4444' : 'var(--text-muted)' }}>
          {driftActive 
            ? 'Bias drift in progress - watch the fairness metric!' 
            : 'Simulates sudden demographic shift to test auto-correction'}
        </p>
      </div>
    </div>
  );
}
