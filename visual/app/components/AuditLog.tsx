"use client";

import { FileText, ChevronRight, User, UserCheck } from "lucide-react";

interface AuditEntry {
  id: number;
  timestamp: string;
  base_prediction: number;
  final_decision: number;
  intervention_type: string;
  protected_value: number;
  true_label: number | null;
}

interface AuditLogProps {
  entries: AuditEntry[];
  onSelectEntry: (id: number) => void;
}

export default function AuditLog({ entries, onSelectEntry }: AuditLogProps) {
  const formatTimestamp = (ts: string) => {
    const date = new Date(ts);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const getDecisionBadge = (decision: number) => {
    if (decision === 1) {
      return <span className="badge badge-success">Approved</span>;
    }
    return <span className="badge badge-danger">Denied</span>;
  };

  const getInterventionBadge = (type: string) => {
    switch (type) {
      case "OVERRIDE_TO_APPROVE":
        return <span className="badge badge-info">↑ Override</span>;
      case "OVERRIDE_TO_DENY":
        return <span className="badge badge-warning">↓ Override</span>;
      case "ACCEPTED":
        return <span className="badge badge-neutral">Accepted</span>;
      case "FAIRFLOW_DISABLED":
        return <span className="badge badge-neutral">Bypassed</span>;
      default:
        return <span className="badge badge-neutral">{type}</span>;
    }
  };

  const getProtectedValueDisplay = (value: number) => {
    // In Adult Census dataset: 0 = Female, 1 = Male
    if (value === 0) {
      return (
        <div className="flex items-center gap-1" style={{ color: '#ec4899' }}>
          <User className="w-4 h-4" />
          <span>Female</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-1" style={{ color: '#3b82f6' }}>
        <UserCheck className="w-4 h-4" />
        <span>Male</span>
      </div>
    );
  };

  return (
    <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: '0.4s' }}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-indigo-400" />
          <h3 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>Audit Log</h3>
        </div>
        <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          {entries.length} decisions recorded
        </span>
      </div>

      <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
        {entries.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
            <p style={{ color: 'var(--text-secondary)' }}>No predictions yet</p>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Start the simulation to see decisions</p>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Time</th>
                <th>Group</th>
                <th>Base Model</th>
                <th>FairFlow</th>
                <th>Intervention</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {entries.slice().reverse().map((entry) => (
                <tr 
                  key={entry.id} 
                  onClick={() => onSelectEntry(entry.id)}
                >
                  <td className="font-mono text-sm" style={{ color: 'var(--text-secondary)' }}>
                    #{entry.id}
                  </td>
                  <td className="text-sm" style={{ color: 'var(--text-primary)' }}>
                    {formatTimestamp(entry.timestamp)}
                  </td>
                  <td>
                    {getProtectedValueDisplay(entry.protected_value)}
                  </td>
                  <td>
                    {getDecisionBadge(entry.base_prediction)}
                  </td>
                  <td>
                    {getDecisionBadge(entry.final_decision)}
                  </td>
                  <td>
                    {getInterventionBadge(entry.intervention_type)}
                  </td>
                  <td>
                    <ChevronRight className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
