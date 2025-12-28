"use client";

import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  icon?: React.ReactNode;
  color?: 'green' | 'blue' | 'yellow' | 'red';
  isFair?: boolean;
}

export default function MetricCard({
  title,
  value,
  subtitle,
  trend,
  trendValue,
  icon,
  color = 'blue',
  isFair,
}: MetricCardProps) {
  const colorClasses = {
    green: 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/30',
    blue: 'from-blue-500/20 to-blue-500/5 border-blue-500/30',
    yellow: 'from-amber-500/20 to-amber-500/5 border-amber-500/30',
    red: 'from-red-500/20 to-red-500/5 border-red-500/30',
  };

  const textColors = {
    green: 'text-emerald-400',
    blue: 'text-blue-400',
    yellow: 'text-amber-400',
    red: 'text-red-400',
  };

  const trendIcons = {
    up: (
      <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
      </svg>
    ),
    down: (
      <svg className="w-4 h-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
      </svg>
    ),
    neutral: (
      <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
      </svg>
    ),
  };

  return (
    <div
      className={`
        relative overflow-hidden rounded-2xl p-6
        bg-gradient-to-br ${colorClasses[color]}
        border backdrop-blur-sm
        transition-all duration-300 hover:scale-[1.02] hover:shadow-lg
        animate-slide-up
      `}
    >
      {/* Background Glow */}
      <div
        className={`absolute -top-10 -right-10 w-32 h-32 rounded-full blur-3xl opacity-20
          ${color === 'green' ? 'bg-emerald-500' : ''}
          ${color === 'blue' ? 'bg-blue-500' : ''}
          ${color === 'yellow' ? 'bg-amber-500' : ''}
          ${color === 'red' ? 'bg-red-500' : ''}
        `}
      />

      <div className="relative">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-slate-400 uppercase tracking-wide">
            {title}
          </span>
          {icon && <span className={textColors[color]}>{icon}</span>}
        </div>

        <div className="flex items-baseline gap-2">
          <span className={`text-4xl font-bold ${textColors[color]}`}>
            {typeof value === 'number' ? value.toFixed(2) : value}
          </span>
          {isFair !== undefined && (
            <span className={`status-badge ${isFair ? 'fair' : 'unfair'}`}>
              {isFair ? '✓ Fair' : '✗ Unfair'}
            </span>
          )}
        </div>

        {(subtitle || trend) && (
          <div className="flex items-center gap-2 mt-2">
            {trend && trendIcons[trend]}
            {trendValue && (
              <span className={`text-sm ${trend === 'up' ? 'text-emerald-400' : trend === 'down' ? 'text-red-400' : 'text-slate-400'}`}>
                {trendValue}
              </span>
            )}
            {subtitle && <span className="text-sm text-slate-500">{subtitle}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
