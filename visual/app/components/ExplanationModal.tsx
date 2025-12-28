"use client";

import { X, AlertTriangle, CheckCircle, XCircle, ArrowRight } from "lucide-react";

interface Contribution {
  feature: string;
  value: number;
  contribution: number;
}

interface ExplanationData {
  id: number;
  prediction: number;
  probability_approve: number;
  intervention_type: string;
  intervention_reason: string;
  detailed_reason: string;
  top_contributors: Contribution[];
  waterfall_plot: string;
}

interface ExplanationModalProps {
  explanation: ExplanationData | null;
  isOpen: boolean;
  onClose: () => void;
  isLoading: boolean;
}

export default function ExplanationModal({
  explanation,
  isOpen,
  onClose,
  isLoading,
}: ExplanationModalProps) {
  if (!isOpen) return null;

  const getContributionBar = (contribution: number, maxContribution: number) => {
    const percentage = Math.abs(contribution / maxContribution) * 100;
    const isPositive = contribution >= 0;
    
    return (
      <div className="flex items-center gap-2 w-full">
        <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-secondary)' }}>
          {isPositive ? (
            <div 
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(percentage, 100)}%`, backgroundColor: '#10b981' }}
            />
          ) : (
            <div className="h-full flex justify-end">
              <div 
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.min(percentage, 100)}%`, backgroundColor: '#ef4444' }}
              />
            </div>
          )}
        </div>
        <span className="text-sm font-mono w-16 text-right" style={{ color: isPositive ? '#10b981' : '#ef4444' }}>
          {isPositive ? '+' : ''}{contribution.toFixed(3)}
        </span>
      </div>
    );
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div 
        className="modal-content w-full max-w-3xl mx-4 p-6"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
              SHAP Explanation
            </h2>
            <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              Understanding why FairFlow made this decision
            </p>
          </div>
          <button 
            onClick={onClose}
            className="p-2 rounded-lg transition-colors"
            style={{ color: 'var(--text-secondary)' }}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {isLoading ? (
          <div className="space-y-4">
            <div className="skeleton h-24 w-full" />
            <div className="skeleton h-48 w-full" />
            <div className="skeleton h-32 w-full" />
          </div>
        ) : explanation ? (
          <div className="space-y-6">
            {/* Decision Summary */}
            <div className="glass-card-static p-4">
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-xl ${
                  explanation.prediction === 1 
                    ? 'bg-emerald-500/20' 
                    : 'bg-red-500/20'
                }`}>
                  {explanation.prediction === 1 ? (
                    <CheckCircle className="w-6 h-6" style={{ color: '#10b981' }} />
                  ) : (
                    <XCircle className="w-6 h-6" style={{ color: '#ef4444' }} />
                  )}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
                      Decision #{explanation.id}
                    </span>
                    <ArrowRight className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
                    <span className="font-semibold" style={{ color: explanation.prediction === 1 ? '#10b981' : '#ef4444' }}>
                      {explanation.prediction === 1 ? 'APPROVED' : 'DENIED'}
                    </span>
                  </div>
                  <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                    Approval Probability: {(explanation.probability_approve * 100).toFixed(1)}%
                  </p>
                </div>
                <div className={`badge ${
                  explanation.intervention_type.includes('OVERRIDE') 
                    ? 'badge-warning' 
                    : 'badge-neutral'
                }`}>
                  {explanation.intervention_type.replace(/_/g, ' ')}
                </div>
              </div>
            </div>

            {/* Intervention Reason */}
            {explanation.intervention_type.includes('OVERRIDE') && (
              <div className="glass-card-static p-4" style={{ borderLeft: '4px solid #f59e0b' }}>
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: '#f59e0b' }} />
                  <div>
                    <p className="font-medium mb-1" style={{ color: 'var(--text-primary)' }}>
                      {explanation.intervention_reason}
                    </p>
                    <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      {explanation.detailed_reason}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Feature Contributions */}
            <div>
              <h3 className="font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
                Top Feature Contributions
              </h3>
              <div className="space-y-3">
                {explanation.top_contributors.map((contrib, idx) => {
                  const maxContrib = Math.max(
                    ...explanation.top_contributors.map(c => Math.abs(c.contribution))
                  );
                  return (
                    <div key={idx} className="glass-card-static p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                          {contrib.feature}
                        </span>
                        <span className="text-sm font-mono" style={{ color: 'var(--text-secondary)' }}>
                          = {Number(contrib.value).toFixed(2)}
                        </span>
                      </div>
                      {getContributionBar(contrib.contribution, maxContrib)}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* SHAP Waterfall Plot */}
            {explanation.waterfall_plot && (
              <div>
                <h3 className="font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
                  SHAP Waterfall Plot
                </h3>
                <div className="glass-card-static p-4 flex justify-center">
                  <img 
                    src={`data:image/png;base64,${explanation.waterfall_plot}`}
                    alt="SHAP Waterfall Plot"
                    className="max-w-full h-auto rounded-lg"
                  />
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-12">
            <AlertTriangle className="w-12 h-12 mx-auto mb-4" style={{ color: '#f59e0b' }} />
            <p style={{ color: 'var(--text-primary)' }}>Failed to load explanation</p>
            <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Please try again</p>
          </div>
        )}

        {/* Footer */}
        <div className="mt-6 pt-6 flex justify-end" style={{ borderTop: '1px solid var(--border-subtle)' }}>
          <button onClick={onClose} className="btn-outline">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
