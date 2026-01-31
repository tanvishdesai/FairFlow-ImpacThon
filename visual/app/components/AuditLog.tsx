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
  features?: Record<string, any>;
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

  const getProtectedValueDisplay = (value: number, features?: Record<string, any>) => {
    // If we have explicit Gender feature, use it
    if (features?.Gender) {
        if (features.Gender === "Male") {
            return (
                <div className="flex items-center gap-1" style={{ color: '#3b82f6' }}>
                  <UserCheck className="w-4 h-4" />
                  <span className="text-xs font-medium">Male</span>
                </div>
            );
        } else if (features.Gender === "Female") {
            return (
                <div className="flex items-center gap-1" style={{ color: '#ec4899' }}>
                  <User className="w-4 h-4" />
                  <span className="text-xs font-medium">Female</span>
                </div>
            );
        } else {
             return (
                <div className="flex items-center gap-1" style={{ color: '#8b5cf6' }}>
                  <User className="w-4 h-4" />
                  <span className="text-xs font-medium">{features.Gender}</span>
                </div>
            );
        }
    }

    // Fallback based on integer value (Legacy / Default)
    // 0 = Unprivileged (Female/Other), 1 = Privileged (Male)
    if (value === 0) {
      return (
        <div className="flex items-center gap-1" style={{ color: '#ec4899' }}>
          <User className="w-4 h-4" />
          <span className="text-xs font-medium">Unprivileged</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-1" style={{ color: '#3b82f6' }}>
        <UserCheck className="w-4 h-4" />
        <span className="text-xs font-medium">Privileged</span>
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
                <>
                <tr 
                  key={entry.id} 
                  onClick={() => onSelectEntry(entry.id)}
                  className="cursor-pointer hover:bg-slate-50/5 relative"
                >
                  <td className="font-mono text-sm" style={{ color: 'var(--text-secondary)' }}>
                    #{entry.id}
                  </td>
                  <td className="text-sm" style={{ color: 'var(--text-primary)' }}>
                    {formatTimestamp(entry.timestamp)}
                  </td>
                  <td>
                    {getProtectedValueDisplay(entry.protected_value, entry.features)}
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
                {/* Expanded Details Row */}
                {entry.features && (
                  <tr>
                    <td colSpan={7} className="p-0 border-0">
                      <div className="bg-slate-50/50 dark:bg-slate-800/30 px-6 py-3 border-b border-slate-100 dark:border-slate-700/50 text-xs">
                        <div className="grid grid-cols-4 gap-4">
                           {entry.features.Job_Role_Applied && (
                              <div className="col-span-1">
                                <span className="text-gray-500 uppercase tracking-wider text-[10px] block mb-1">Role</span>
                                <span className="font-medium text-slate-700 dark:text-slate-300">{entry.features.Job_Role_Applied}</span>
                              </div>
                           )}
                           {entry.features.Education_Level && (
                              <div className="col-span-1">
                                <span className="text-gray-500 uppercase tracking-wider text-[10px] block mb-1">Education</span>
                                <span className="font-medium text-slate-700 dark:text-slate-300">{entry.features.Education_Level}</span>
                              </div>
                           )}
                           {entry.features.Experience_Years !== undefined && (
                              <div className="col-span-1">
                                <span className="text-gray-500 uppercase tracking-wider text-[10px] block mb-1">Experience</span>
                                <span className="font-medium text-slate-700 dark:text-slate-300">{entry.features.Experience_Years} Years</span>
                              </div>
                           )}
                           {entry.features.Skill_Score !== undefined && (
                              <div className="col-span-1">
                                <span className="text-gray-500 uppercase tracking-wider text-[10px] block mb-1">Skill Score</span>
                                <span className="font-medium text-slate-700 dark:text-slate-300">{entry.features.Skill_Score}/100</span>
                              </div>
                           )}
                           {/* Fallback for Generic/Other datasets */}
                           {!entry.features.Job_Role_Applied && Object.keys(entry.features).slice(0, 4).map((key) => (
                             <div key={key} className="col-span-1">
                                <span className="text-gray-500 uppercase tracking-wider text-[10px] block mb-1">{key}</span>
                                <span className="font-medium text-slate-700 dark:text-slate-300">{JSON.stringify(entry.features?.[key])}</span>
                             </div>
                           ))}
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
                </>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
