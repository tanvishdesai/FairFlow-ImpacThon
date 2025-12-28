"use client";

import React from 'react';

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

interface ShapExplanationModalProps {
  data: ExplanationData | null;
  isOpen: boolean;
  onClose: () => void;
  loading?: boolean;
}

export default function ShapExplanationModal({
  data,
  isOpen,
  onClose,
  loading,
}: ShapExplanationModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto glass-card p-0">
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between p-6 border-b border-white/10 bg-slate-900/90 backdrop-blur-sm">
          <div>
            <h2 className="text-2xl font-bold text-white">
              Prediction Explanation
            </h2>
            {data && (
              <p className="text-sm text-slate-400 mt-1">
                ID: #{data.id} • {data.intervention_type}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <svg className="w-6 h-6 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {loading ? (
            <div className="flex items-center justify-center h-64">
              <svg className="w-12 h-12 animate-spin text-slate-400" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            </div>
          ) : data ? (
            <>
              {/* Intervention Summary */}
              <div className={`
                p-4 rounded-xl mb-6 border
                ${data.intervention_type.includes('OVERRIDE')
                  ? 'bg-purple-500/10 border-purple-500/30'
                  : 'bg-slate-500/10 border-slate-500/30'
                }
              `}>
                <div className="flex items-start gap-3">
                  <div className={`
                    p-2 rounded-lg
                    ${data.intervention_type.includes('OVERRIDE')
                      ? 'bg-purple-500/20 text-purple-400'
                      : 'bg-slate-500/20 text-slate-400'
                    }
                  `}>
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="font-semibold text-white mb-1">
                      {data.intervention_reason}
                    </h3>
                    <p className="text-sm text-slate-400">
                      {data.detailed_reason}
                    </p>
                  </div>
                </div>
              </div>

              {/* Prediction Details */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="p-4 rounded-xl bg-slate-800/50">
                  <p className="text-sm text-slate-400 mb-1">Final Decision</p>
                  <p className={`text-2xl font-bold ${
                    data.prediction === 1 ? 'text-emerald-400' : 'text-red-400'
                  }`}>
                    {data.prediction === 1 ? 'APPROVED' : 'DENIED'}
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-800/50">
                  <p className="text-sm text-slate-400 mb-1">Approval Probability</p>
                  <p className="text-2xl font-bold text-blue-400">
                    {(data.probability_approve * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* SHAP Contributors */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-4">
                  Top Feature Contributors
                </h3>
                <div className="space-y-3">
                  {data.top_contributors.slice(0, 8).map((contrib, idx) => (
                    <div key={idx} className="flex items-center gap-4">
                      <div className="w-32 text-sm text-slate-400 truncate" title={contrib.feature}>
                        {contrib.feature}
                      </div>
                      <div className="flex-1 relative h-6">
                        <div className="absolute inset-y-0 left-1/2 w-px bg-slate-600" />
                        <div
                          className={`absolute top-0 h-full rounded ${
                            contrib.shap_value >= 0
                              ? 'left-1/2 bg-emerald-500/60'
                              : 'right-1/2 bg-red-500/60'
                          }`}
                          style={{
                            width: `${Math.min(Math.abs(contrib.shap_value) * 100, 50)}%`,
                          }}
                        />
                      </div>
                      <div className={`w-16 text-right text-sm font-mono ${
                        contrib.shap_value >= 0 ? 'text-emerald-400' : 'text-red-400'
                      }`}>
                        {contrib.shap_value >= 0 ? '+' : ''}{contrib.shap_value.toFixed(3)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* SHAP Waterfall Plot */}
              {data.waterfall_plot && (
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">
                    SHAP Waterfall Plot
                  </h3>
                  <div className="rounded-xl overflow-hidden bg-white p-2">
                    <img
                      src={`data:image/png;base64,${data.waterfall_plot}`}
                      alt="SHAP Waterfall Plot"
                      className="w-full"
                    />
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12 text-slate-400">
              <p>No explanation data available</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
