"use client";

import React from 'react';

interface ActivationToggleProps {
  active: boolean;
  onToggle: (active: boolean) => void;
  loading?: boolean;
}

export default function ActivationToggle({ active, onToggle, loading }: ActivationToggleProps) {
  return (
    <div className="glass-card p-8 flex flex-col items-center">
      <h2 className="text-xl font-bold text-white mb-2">FairFlow Control</h2>
      <p className="text-sm text-slate-400 mb-6 text-center">
        Toggle to enable/disable real-time bias mitigation
      </p>

      {/* Large Toggle Button */}
      <button
        onClick={() => onToggle(!active)}
        disabled={loading}
        className={`
          toggle-switch relative w-40 h-40 rounded-full
          flex items-center justify-center
          transition-all duration-500 ease-out
          ${active 
            ? 'bg-gradient-to-br from-emerald-500 to-emerald-700 glow-green active' 
            : 'bg-gradient-to-br from-slate-700 to-slate-800'
          }
          ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:scale-105'}
          border-4 ${active ? 'border-emerald-400' : 'border-slate-600'}
        `}
      >
        {/* Inner Ring */}
        <div
          className={`
            absolute inset-3 rounded-full
            ${active ? 'bg-emerald-600/30' : 'bg-slate-700/50'}
            transition-all duration-500
          `}
        />

        {/* Icon/Text */}
        <div className="relative z-10 flex flex-col items-center">
          {loading ? (
            <svg className="w-12 h-12 animate-spin text-white" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : active ? (
            <>
              <svg className="w-12 h-12 text-white mb-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              <span className="text-sm font-bold text-white">ACTIVE</span>
            </>
          ) : (
            <>
              <svg className="w-12 h-12 text-slate-400 mb-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.618 5.984A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016zM12 9v2m0 4h.01" />
              </svg>
              <span className="text-sm font-medium text-slate-400">INACTIVE</span>
            </>
          )}
        </div>

        {/* Pulse Ring when Active */}
        {active && !loading && (
          <div className="absolute inset-0 rounded-full border-4 border-emerald-400 animate-ping opacity-20" />
        )}
      </button>

      {/* Status Text */}
      <div className="mt-6 text-center">
        <p className={`text-lg font-semibold ${active ? 'text-emerald-400' : 'text-slate-500'}`}>
          {active ? 'Bias mitigation enabled' : 'Click to activate protection'}
        </p>
        {active && (
          <p className="text-xs text-slate-500 mt-1">
            RL Gatekeeper is monitoring all predictions
          </p>
        )}
      </div>
    </div>
  );
}
