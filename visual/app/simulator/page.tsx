"use client";

import { useState, useEffect } from "react";
import { 
  User, 
  CheckCircle,
  XCircle,
  AlertTriangle,
  Sparkles,
  Scale,
  TrendingUp,
  Users,
  ArrowRight,
  RefreshCw
} from "lucide-react";
import Link from "next/link";

const API_BASE = "http://localhost:8000";

// Types for precomputed results
interface DemoCase {
  type: string;
  display_name: string;
  description: string;
  row_index: number;
  gender: string;
  true_label: number;
  true_label_text: string;
  base_prediction: number;
  base_prediction_text: string;
  base_probability: number;
  fairflow_decision: number;
  fairflow_decision_text: string;
  intervention_occurred: boolean;
  intervention_type: string;
  current_dpr_at_decision: number;
  is_model_correct: boolean;
  result_type: string;
  features?: {
    Age: number;
    Job: number;
    Housing: number;
    Saving_accounts: number;
    Checking_account: number;
    Credit_amount: number;
    Duration: number;
    Purpose: number;
  };
}

interface ModelStats {
  accuracy: number;
  male_approval_rate: number;
  female_approval_rate: number;
  dpr: number;
}

interface FairFlowStats extends ModelStats {
  final_dpr: number;
  total_interventions: number;
}

interface PrecomputedData {
  base_model_stats: ModelStats;
  fairflow_stats: FairFlowStats;
  demo_cases: DemoCase[];
}

export default function SimulatorPage() {
  const [data, setData] = useState<PrecomputedData | null>(null);
  const [selectedCase, setSelectedCase] = useState<DemoCase | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch precomputed demo cases on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/precomputed-demo-cases`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const result = await res.json();
        setData(result);
        // Auto-select first bias victim case
        if (result.demo_cases?.length > 0) {
          const biasVictim = result.demo_cases.find((c: DemoCase) => c.type === "bias_victim");
          setSelectedCase(biasVictim || result.demo_cases[0]);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load demo cases");
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--bg-primary)' }}>
        <div className="text-center">
          <RefreshCw className="w-12 h-12 animate-spin mx-auto mb-4" style={{ color: '#6366f1' }} />
          <p style={{ color: 'var(--text-secondary)' }}>Loading precomputed demo cases...</p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--bg-primary)' }}>
        <div className="text-center glass-card p-8">
          <XCircle className="w-12 h-12 mx-auto mb-4" style={{ color: '#ef4444' }} />
          <p className="text-lg mb-2" style={{ color: 'var(--text-primary)' }}>Error Loading Demo</p>
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{error}</p>
          <p className="text-xs mt-4" style={{ color: 'var(--text-muted)' }}>
            Run: python scripts/precompute_examples.py
          </p>
        </div>
      </div>
    );
  }

  const getCaseColor = (type: string) => {
    switch (type) {
      case "bias_victim": return "#ef4444";
      case "correct_denial": return "#10b981";
      case "male_baseline": return "#3b82f6";
      default: return "#6366f1";
    }
  };

  const getCaseIcon = (type: string) => {
    switch (type) {
      case "bias_victim": return "🔴";
      case "correct_denial": return "🟢";
      case "male_baseline": return "🔵";
      default: return "⚪";
    }
  };

  return (
    <div className="min-h-screen p-6" style={{ background: 'var(--bg-primary)' }}>
      <div className="max-w-[1400px] mx-auto">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <Link href="/" className="text-sm hover:underline" style={{ color: 'var(--text-secondary)' }}>
                ← Back to Dashboard
              </Link>
              <h1 className="text-3xl font-bold mt-2" style={{ color: 'var(--text-primary)' }}>
                <Scale className="inline-block mr-3 w-8 h-8" style={{ color: '#6366f1' }} />
                Individual Case Simulator
              </h1>
              <p className="mt-2" style={{ color: 'var(--text-secondary)' }}>
                Real examples from German Credit test set showing FairFlow's bias correction
              </p>
            </div>
          </div>
        </header>

        {/* Stats Overview */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="glass-card p-4 text-center">
            <div className="text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>Base Model DPR</div>
            <div className="text-3xl font-bold" style={{ color: '#ef4444' }}>
              {data.base_model_stats.dpr.toFixed(2)}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Below threshold (0.80)</div>
          </div>
          <div className="glass-card p-4 text-center">
            <div className="text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>FairFlow DPR</div>
            <div className="text-3xl font-bold" style={{ color: '#10b981' }}>
              {data.fairflow_stats.final_dpr.toFixed(2)}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Fair &amp; equitable</div>
          </div>
          <div className="glass-card p-4 text-center">
            <div className="text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>Total Interventions</div>
            <div className="text-3xl font-bold" style={{ color: '#6366f1' }}>
              {data.fairflow_stats.total_interventions}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Bias corrections</div>
          </div>
          <div className="glass-card p-4 text-center">
            <div className="text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>Approval Gap Fixed</div>
            <div className="text-3xl font-bold" style={{ color: '#f59e0b' }}>
              {((data.base_model_stats.male_approval_rate - data.base_model_stats.female_approval_rate) * 100).toFixed(0)}%
            </div>
            <div className="text-xs" style={{ color: 'var(--text-muted)' }}>M: {(data.base_model_stats.male_approval_rate * 100).toFixed(0)}% vs F: {(data.base_model_stats.female_approval_rate * 100).toFixed(0)}%</div>
          </div>
        </div>

        {/* Demo Cases Selector */}
        <div className="glass-card p-4 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="w-5 h-5" style={{ color: '#f59e0b' }} />
            <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
              Select a Demo Case
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {data.demo_cases.map((demoCase, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedCase(demoCase)}
                className={`p-4 rounded-lg text-left transition-all ${
                  selectedCase?.row_index === demoCase.row_index 
                    ? 'ring-2 ring-indigo-500' 
                    : 'hover:bg-white/5'
                }`}
                style={{ 
                  backgroundColor: selectedCase?.row_index === demoCase.row_index 
                    ? 'rgba(99,102,241,0.15)' 
                    : 'var(--bg-secondary)',
                  borderLeft: `4px solid ${getCaseColor(demoCase.type)}`
                }}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span>{getCaseIcon(demoCase.type)}</span>
                  <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                    {demoCase.display_name}
                  </span>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  {demoCase.description}
                </p>
              </button>
            ))}
          </div>
        </div>

        {/* Selected Case Details */}
        {selectedCase && (
          <div className="space-y-6">
            {/* Applicant Info */}
            <div className="glass-card p-4">
              <div className="flex items-center gap-4">
                <User className="w-8 h-8" style={{ color: selectedCase.gender === 'Female' ? '#ec4899' : '#3b82f6' }} />
                <div>
                  <h3 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                    {selectedCase.display_name}
                  </h3>
                  <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                    {selectedCase.gender} applicant • Applicant #{selectedCase.row_index} in test set • 
                    True creditworthiness: <strong style={{ color: selectedCase.true_label === 1 ? '#10b981' : '#ef4444' }}>
                      {selectedCase.true_label_text}
                    </strong>
                  </p>
                </div>
              </div>
            </div>

            {/* Feature Values - Explains why the model made its decision */}
            {selectedCase.features && (
              <div className="glass-card p-4">
                <h4 className="font-semibold mb-3 flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
                  <TrendingUp className="w-4 h-4" style={{ color: '#6366f1' }} />
                  Applicant Features (Normalized Values)
                </h4>
                <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                  These scaled feature values are what the model used for its decision. Positive values indicate above-average, negative below-average.
                </p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Age</div>
                    <div className="text-lg font-bold" style={{ color: selectedCase.features.Age > 0 ? '#10b981' : '#ef4444' }}>
                      {selectedCase.features.Age > 0 ? '+' : ''}{selectedCase.features.Age.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Job Level</div>
                    <div className="text-lg font-bold" style={{ color: selectedCase.features.Job > 0 ? '#10b981' : '#ef4444' }}>
                      {selectedCase.features.Job > 0 ? '+' : ''}{selectedCase.features.Job.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Savings</div>
                    <div className="text-lg font-bold" style={{ color: selectedCase.features.Saving_accounts > 0 ? '#10b981' : '#ef4444' }}>
                      {selectedCase.features.Saving_accounts > 0 ? '+' : ''}{selectedCase.features.Saving_accounts.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Checking</div>
                    <div className="text-lg font-bold" style={{ color: selectedCase.features.Checking_account > 0 ? '#10b981' : '#ef4444' }}>
                      {selectedCase.features.Checking_account > 0 ? '+' : ''}{selectedCase.features.Checking_account.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Credit Amount</div>
                    <div className="text-lg font-bold" style={{ color: selectedCase.features.Credit_amount > 0 ? '#f59e0b' : '#10b981' }}>
                      {selectedCase.features.Credit_amount > 0 ? '+' : ''}{selectedCase.features.Credit_amount.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Duration</div>
                    <div className="text-lg font-bold" style={{ color: selectedCase.features.Duration > 0 ? '#f59e0b' : '#10b981' }}>
                      {selectedCase.features.Duration > 0 ? '+' : ''}{selectedCase.features.Duration.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Housing</div>
                    <div className="text-lg font-bold" style={{ color: selectedCase.features.Housing > 0 ? '#10b981' : '#ef4444' }}>
                      {selectedCase.features.Housing > 0 ? '+' : ''}{selectedCase.features.Housing.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-center p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Purpose</div>
                    <div className="text-lg font-bold" style={{ color: '#6366f1' }}>
                      {selectedCase.features.Purpose.toFixed(2)}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Three-Panel Decision Flow */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* True Label Panel */}
              <div 
                className="glass-card p-6 relative overflow-hidden"
                style={{ 
                  borderColor: selectedCase.true_label === 1 ? '#10b981' : '#ef4444',
                  borderWidth: '2px'
                }}
              >
                <div className="absolute top-0 left-0 right-0 h-1" style={{ 
                  backgroundColor: selectedCase.true_label === 1 ? '#10b981' : '#ef4444' 
                }} />
                <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                  GROUND TRUTH
                </h3>
                <div className="flex items-center gap-3 mb-3">
                  {selectedCase.true_label === 1 ? (
                    <CheckCircle className="w-12 h-12" style={{ color: '#10b981' }} />
                  ) : (
                    <XCircle className="w-12 h-12" style={{ color: '#ef4444' }} />
                  )}
                  <div>
                    <div className="text-2xl font-bold" style={{ 
                      color: selectedCase.true_label === 1 ? '#10b981' : '#ef4444' 
                    }}>
                      {selectedCase.true_label_text}
                    </div>
                    <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      Actual creditworthiness
                    </div>
                  </div>
                </div>
              </div>

              {/* Base Model Panel */}
              <div 
                className="glass-card p-6 relative overflow-hidden"
                style={{ 
                  borderColor: selectedCase.is_model_correct ? '#10b981' : '#ef4444',
                  borderWidth: '2px'
                }}
              >
                <div className="absolute top-0 left-0 right-0 h-1" style={{ 
                  backgroundColor: selectedCase.base_prediction === 1 ? '#10b981' : '#ef4444' 
                }} />
                <h3 className="text-sm font-medium mb-4 flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
                  BIASED MODEL
                  {!selectedCase.is_model_correct && (
                    <span className="px-2 py-0.5 text-xs rounded-full" style={{ backgroundColor: 'rgba(239,68,68,0.2)', color: '#ef4444' }}>
                      WRONG
                    </span>
                  )}
                </h3>
                <div className="flex items-center gap-3 mb-3">
                  {selectedCase.base_prediction === 1 ? (
                    <CheckCircle className="w-12 h-12" style={{ color: '#10b981' }} />
                  ) : (
                    <XCircle className="w-12 h-12" style={{ color: '#ef4444' }} />
                  )}
                  <div>
                    <div className="text-2xl font-bold" style={{ 
                      color: selectedCase.base_prediction === 1 ? '#10b981' : '#ef4444' 
                    }}>
                      {selectedCase.base_prediction_text}
                    </div>
                    <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      {(selectedCase.base_probability * 100).toFixed(0)}% confidence
                    </div>
                  </div>
                </div>
              </div>

              {/* FairFlow Panel */}
              <div 
                className={`glass-card p-6 relative overflow-hidden ${selectedCase.intervention_occurred ? 'animate-pulse-once' : ''}`}
                style={{ 
                  borderColor: selectedCase.intervention_occurred ? '#6366f1' : (selectedCase.fairflow_decision === 1 ? '#10b981' : '#ef4444'),
                  borderWidth: '2px',
                  boxShadow: selectedCase.intervention_occurred ? '0 0 20px rgba(99,102,241,0.3)' : undefined
                }}
              >
                <div className="absolute top-0 left-0 right-0 h-1" style={{ 
                  backgroundColor: selectedCase.intervention_occurred ? '#6366f1' : (selectedCase.fairflow_decision === 1 ? '#10b981' : '#ef4444')
                }} />
                <h3 className="text-sm font-medium mb-4 flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
                  FAIRFLOW
                  {selectedCase.intervention_occurred && (
                    <span className="px-2 py-0.5 text-xs rounded-full" style={{ backgroundColor: 'rgba(99,102,241,0.2)', color: '#6366f1' }}>
                      OVERRIDE
                    </span>
                  )}
                </h3>
                <div className="flex items-center gap-3 mb-3">
                  {selectedCase.fairflow_decision === 1 ? (
                    <CheckCircle className="w-12 h-12" style={{ color: '#10b981' }} />
                  ) : (
                    <XCircle className="w-12 h-12" style={{ color: '#ef4444' }} />
                  )}
                  <div>
                    <div className="text-2xl font-bold" style={{ 
                      color: selectedCase.fairflow_decision === 1 ? '#10b981' : '#ef4444' 
                    }}>
                      {selectedCase.fairflow_decision_text}
                    </div>
                    <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      {selectedCase.intervention_type.replace(/_/g, ' ')}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Intervention Explanation */}
            {selectedCase.intervention_occurred && (
              <div 
                className="glass-card p-5 border-l-4"
                style={{ borderLeftColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.05)' }}
              >
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-6 h-6 flex-shrink-0 mt-0.5" style={{ color: '#f59e0b' }} />
                  <div>
                    <h4 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
                      Why FairFlow Intervened
                    </h4>
                    <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                      The base model <strong>DENIED</strong> this {selectedCase.gender.toLowerCase()} applicant despite 
                      being <strong style={{ color: '#10b981' }}>CREDITWORTHY</strong>. 
                      At the time of decision, the DPR was <strong>{selectedCase.current_dpr_at_decision.toFixed(3)}</strong> (below 0.80 threshold), 
                      indicating unfair treatment of the unprivileged group. 
                      FairFlow overrode the decision to <strong style={{ color: '#10b981' }}>APPROVE</strong> to restore fairness.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* No Intervention Explanation */}
            {!selectedCase.intervention_occurred && selectedCase.type === "correct_denial" && (
              <div 
                className="glass-card p-5 border-l-4"
                style={{ borderLeftColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.05)' }}
              >
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-6 h-6 flex-shrink-0 mt-0.5" style={{ color: '#10b981' }} />
                  <div>
                    <h4 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
                      Why FairFlow Did NOT Intervene
                    </h4>
                    <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                      The base model correctly <strong>DENIED</strong> this applicant who is truly 
                      <strong style={{ color: '#ef4444' }}> HIGH-RISK</strong>. 
                      FairFlow only corrects <em>unfair</em> denials, not legitimate risk-based decisions.
                      This demonstrates that FairFlow doesn't blindly approve all unprivileged applicants.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Legend */}
            <div className="glass-card p-4">
              <h4 className="font-medium mb-3" style={{ color: 'var(--text-primary)' }}>Case Types</h4>
              <div className="flex flex-wrap gap-6">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: '#ef4444' }} />
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                    Bias Victim: Creditworthy but wrongly denied → FairFlow overrides
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: '#10b981' }} />
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                    Correct Denial: High-risk correctly denied → No intervention needed
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: '#3b82f6' }} />
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                    Male Baseline: Shows model's favorable treatment of privileged group
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-8 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
          <p>FairFlow: The RL-Driven Adaptive Bias Firewall</p>
          <p className="mt-1 text-xs">Using precomputed results from 200 test samples with accumulated DPR context</p>
        </footer>
      </div>

      <style jsx>{`
        @keyframes pulse-once {
          0%, 100% { box-shadow: 0 0 20px rgba(99,102,241,0); }
          50% { box-shadow: 0 0 30px rgba(99,102,241,0.4); }
        }
        .animate-pulse-once {
          animation: pulse-once 1s ease-in-out;
        }
      `}</style>
    </div>
  );
}
