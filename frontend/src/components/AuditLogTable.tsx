"use client";

import React from 'react';

interface AuditEntry {
  id: number;
  timestamp: string;
  base_prediction: number;
  final_decision: number;
  intervention_type: string;
  protected_value: number;
  true_label: number | null;
}

interface AuditLogTableProps {
  entries: AuditEntry[];
  onRowClick?: (entry: AuditEntry) => void;
  loading?: boolean;
}

export default function AuditLogTable({ entries, onRowClick, loading }: AuditLogTableProps) {
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const getDecisionBadge = (decision: number) => {
    return decision === 1 ? (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
        Approve
      </span>
    ) : (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-red-500/20 text-red-400 border border-red-500/30">
        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
        </svg>
        Deny
      </span>
    );
  };

  const getInterventionBadge = (type: string) => {
    if (type === 'ACCEPTED' || type === 'FAIRFLOW_DISABLED') {
      return (
        <span className="text-xs text-slate-500">—</span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-purple-500/20 text-purple-400 border border-purple-500/30">
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        {type.replace('OVERRIDE_TO_', '')}
      </span>
    );
  };

  if (loading) {
    return (
      <div className="glass-card p-6">
        <h2 className="text-xl font-bold text-white mb-4">Audit Log</h2>
        <div className="flex items-center justify-center h-48">
          <svg className="w-8 h-8 animate-spin text-slate-400" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-white">Audit Log</h2>
        <span className="text-sm text-slate-400">{entries.length} records</span>
      </div>

      <div className="overflow-x-auto max-h-96 overflow-y-auto">
        {entries.length === 0 ? (
          <div className="text-center py-12 text-slate-500">
            <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p>No predictions yet</p>
            <p className="text-sm mt-1">Start the simulation to see audit entries</p>
          </div>
        ) : (
          <table className="audit-table">
            <thead className="sticky top-0 bg-slate-900/90 backdrop-blur-sm">
              <tr>
                <th>ID</th>
                <th>Time</th>
                <th>Base Decision</th>
                <th>Final Decision</th>
                <th>Intervention</th>
                <th>Sex</th>
              </tr>
            </thead>
            <tbody>
              {entries.slice().reverse().map((entry) => {
                const isIntervention = entry.base_prediction !== entry.final_decision;
                return (
                  <tr
                    key={entry.id}
                    className={`cursor-pointer transition-colors ${isIntervention ? 'intervention' : ''}`}
                    onClick={() => onRowClick?.(entry)}
                  >
                    <td className="font-mono text-sm text-slate-400">#{entry.id}</td>
                    <td className="text-sm text-slate-300">{formatTime(entry.timestamp)}</td>
                    <td>{getDecisionBadge(entry.base_prediction)}</td>
                    <td>{getDecisionBadge(entry.final_decision)}</td>
                    <td>{getInterventionBadge(entry.intervention_type)}</td>
                    <td>
                      <span className={`text-xs px-2 py-1 rounded ${
                        entry.protected_value === 1 
                          ? 'bg-blue-500/20 text-blue-400' 
                          : 'bg-pink-500/20 text-pink-400'
                      }`}>
                        {entry.protected_value === 1 ? '♂ Male' : '♀ Female'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
