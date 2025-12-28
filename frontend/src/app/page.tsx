"use client";

import { useState, useEffect, useCallback } from 'react';
import MetricCard from '../components/MetricCard';
import LiveMetricsChart from '../components/LiveMetricsChart';
import ActivationToggle from '../components/ActivationToggle';
import AuditLogTable from '../components/AuditLogTable';
import ShapExplanationModal from '../components/ShapExplanationModal';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Metrics {
  accuracy: number;
  demographic_parity_ratio: number;
  demographic_parity_difference: number;
  privileged_approval_rate: number;
  unprivileged_approval_rate: number;
  is_fair: boolean;
  total_predictions: number;
  total_interventions: number;
  intervention_rate: number;
}

interface AuditEntry {
  id: number;
  timestamp: string;
  base_prediction: number;
  final_decision: number;
  intervention_type: string;
  protected_value: number;
  true_label: number | null;
}

interface ChartDataPoint {
  timestamp: string;
  accuracy: number;
  dpr: number;
  interventionRate: number;
}

interface ExplanationData {
  id: number;
  prediction: number;
  probability_approve: number;
  intervention_type: string;
  intervention_reason: string;
  detailed_reason: string;
  top_contributors: Array<{
    feature: string;
    value: number;
    shap_value: number;
    direction: string;
  }>;
  waterfall_plot: string;
}

export default function Dashboard() {
  // State
  const [fairflowActive, setFairflowActive] = useState(true);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [auditLog, setAuditLog] = useState<AuditEntry[]>([]);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [apiConnected, setApiConnected] = useState(false);

  // Modal state
  const [selectedEntry, setSelectedEntry] = useState<AuditEntry | null>(null);
  const [explanation, setExplanation] = useState<ExplanationData | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);

  // Fetch metrics
  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/metrics`);
      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
        setApiConnected(true);

        // Add to chart data
        setChartData(prev => {
          const newPoint: ChartDataPoint = {
            timestamp: new Date().toISOString(),
            accuracy: data.accuracy,
            dpr: data.demographic_parity_ratio,
            interventionRate: data.intervention_rate,
          };
          const updated = [...prev, newPoint].slice(-50); // Keep last 50 points
          return updated;
        });
      }
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
      setApiConnected(false);
    }
  }, []);

  // Fetch audit log
  const fetchAuditLog = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/audit-log?limit=100`);
      if (response.ok) {
        const data = await response.json();
        setAuditLog(data);
      }
    } catch (error) {
      console.error('Failed to fetch audit log:', error);
    }
  }, []);

  // Fetch simulation status
  const fetchSimulationStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/simulate/status`);
      if (response.ok) {
        const data = await response.json();
        setSimulationRunning(data.is_running);
        setFairflowActive(data.fairflow_active);
      }
    } catch (error) {
      console.error('Failed to fetch simulation status:', error);
    }
  }, []);

  // Toggle FairFlow
  const handleToggleFairFlow = async (active: boolean) => {
    try {
      await fetch(`${API_BASE}/api/fairflow/toggle?active=${active}`, {
        method: 'POST',
      });
      setFairflowActive(active);
    } catch (error) {
      console.error('Failed to toggle FairFlow:', error);
    }
  };

  // Start/Stop simulation
  const handleSimulation = async (action: 'start' | 'stop') => {
    try {
      await fetch(`${API_BASE}/api/simulate/${action}`, {
        method: 'POST',
      });
      setSimulationRunning(action === 'start');
    } catch (error) {
      console.error(`Failed to ${action} simulation:`, error);
    }
  };

  // Inject drift
  const handleInjectDrift = async () => {
    try {
      await fetch(`${API_BASE}/api/simulate/inject-drift`, {
        method: 'POST',
      });
    } catch (error) {
      console.error('Failed to inject drift:', error);
    }
  };

  // Fetch explanation for entry
  const fetchExplanation = async (entry: AuditEntry) => {
    setSelectedEntry(entry);
    setExplanationLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/explain/${entry.id}`);
      if (response.ok) {
        const data = await response.json();
        setExplanation(data);
      }
    } catch (error) {
      console.error('Failed to fetch explanation:', error);
    } finally {
      setExplanationLoading(false);
    }
  };

  // Initial load and polling
  useEffect(() => {
    const init = async () => {
      await Promise.all([fetchMetrics(), fetchAuditLog(), fetchSimulationStatus()]);
      setLoading(false);
    };
    init();

    // Poll for updates
    const interval = setInterval(() => {
      fetchMetrics();
      fetchAuditLog();
      fetchSimulationStatus();
    }, 1000);

    return () => clearInterval(interval);
  }, [fetchMetrics, fetchAuditLog, fetchSimulationStatus]);

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-xl bg-gradient-to-br from-emerald-500 to-blue-600">
              <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-blue-500 bg-clip-text text-transparent">
                FairFlow
              </h1>
              <p className="text-sm text-slate-400">
                RL-Driven Adaptive Bias Firewall
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* API Status */}
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
              apiConnected 
                ? 'bg-emerald-500/20 text-emerald-400' 
                : 'bg-red-500/20 text-red-400'
            }`}>
              <div className={`w-2 h-2 rounded-full ${apiConnected ? 'bg-emerald-500' : 'bg-red-500'} ${apiConnected ? 'animate-pulse' : ''}`} />
              {apiConnected ? 'API Connected' : 'API Disconnected'}
            </div>

            {/* Simulation Controls */}
            <div className="flex items-center gap-2">
              {!simulationRunning ? (
                <button
                  onClick={() => handleSimulation('start')}
                  disabled={!apiConnected}
                  className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition-colors disabled:opacity-50"
                >
                  ▶ Start Simulation
                </button>
              ) : (
                <>
                  <button
                    onClick={() => handleSimulation('stop')}
                    className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white font-medium transition-colors"
                  >
                    ⏹ Stop
                  </button>
                  <button
                    onClick={handleInjectDrift}
                    className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-medium transition-colors"
                  >
                    ⚡ Inject Drift
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Metrics Row */}
        <div className="col-span-12 grid grid-cols-4 gap-4">
          <MetricCard
            title="Accuracy"
            value={metrics?.accuracy ?? 0}
            color="green"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            }
          />
          <MetricCard
            title="Demographic Parity"
            value={metrics?.demographic_parity_ratio ?? 1}
            color={metrics?.is_fair ? 'blue' : 'red'}
            isFair={metrics?.is_fair}
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
              </svg>
            }
          />
          <MetricCard
            title="Intervention Rate"
            value={`${((metrics?.intervention_rate ?? 0) * 100).toFixed(1)}%`}
            subtitle={`${metrics?.total_interventions ?? 0} of ${metrics?.total_predictions ?? 0}`}
            color="yellow"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            }
          />
          <MetricCard
            title="Total Predictions"
            value={metrics?.total_predictions ?? 0}
            color="blue"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
              </svg>
            }
          />
        </div>

        {/* Chart */}
        <div className="col-span-9">
          <LiveMetricsChart data={chartData} fairflowActive={fairflowActive} />
        </div>

        {/* Toggle */}
        <div className="col-span-3">
          <ActivationToggle
            active={fairflowActive}
            onToggle={handleToggleFairFlow}
          />
        </div>

        {/* Audit Log */}
        <div className="col-span-12">
          <AuditLogTable
            entries={auditLog}
            onRowClick={fetchExplanation}
            loading={loading}
          />
        </div>
      </div>

      {/* Explanation Modal */}
      <ShapExplanationModal
        isOpen={selectedEntry !== null}
        data={explanation}
        loading={explanationLoading}
        onClose={() => {
          setSelectedEntry(null);
          setExplanation(null);
        }}
      />
    </div>
  );
}
