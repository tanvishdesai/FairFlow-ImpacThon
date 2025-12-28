"use client";

import { LucideIcon, TrendingUp, TrendingDown, Minus } from "lucide-react";

interface MetricsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  trend?: "up" | "down" | "neutral";
  status?: "good" | "warning" | "danger" | "neutral";
  animate?: boolean;
}

export default function MetricsCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  status = "neutral",
  animate = true,
}: MetricsCardProps) {
  const statusColors = {
    good: {
      iconBg: "from-emerald-500 to-green-600",
      glow: "glow-green",
      textColor: "#10b981", // emerald
    },
    warning: {
      iconBg: "from-amber-500 to-orange-600",
      glow: "",
      textColor: "#f59e0b", // amber
    },
    danger: {
      iconBg: "from-red-500 to-rose-600",
      glow: "glow-red",
      textColor: "#ef4444", // red
    },
    neutral: {
      iconBg: "from-indigo-500 to-purple-600",
      glow: "",
      textColor: "#6366f1", // indigo
    },
  };

  const colors = statusColors[status];

  const TrendIcon = trend === "up" ? TrendingUp : trend === "down" ? TrendingDown : Minus;

  return (
    <div className={`glass-card p-6 ${animate ? 'animate-slide-up' : ''}`} style={{ animationDelay: '0.1s' }}>
      <div className="flex items-start justify-between mb-4">
        <div className={`p-3 rounded-xl bg-gradient-to-br ${colors.iconBg} ${status !== 'neutral' ? colors.glow : ''}`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        {trend && (
          <div className={`flex items-center gap-1`} style={{ 
            color: trend === 'up' ? '#10b981' : trend === 'down' ? '#ef4444' : 'var(--text-secondary)' 
          }}>
            <TrendIcon className="w-4 h-4" />
          </div>
        )}
      </div>
      
      <div>
        <p className="text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>{title}</p>
        <p className="text-3xl font-bold" style={{ color: colors.textColor }}>
          {value}
        </p>
        {subtitle && (
          <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{subtitle}</p>
        )}
      </div>
    </div>
  );
}
