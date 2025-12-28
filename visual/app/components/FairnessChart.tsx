"use client";

import { useEffect, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { TrendingUp } from "lucide-react";

interface DataPoint {
  time: string;
  accuracy: number;
  fairness: number;
}

interface FairnessChartProps {
  data: DataPoint[];
  fairnessThreshold?: number;
}

export default function FairnessChart({ 
  data, 
  fairnessThreshold = 0.8 
}: FairnessChartProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: '0.3s' }}>
        <div className="flex items-center gap-2 mb-6">
          <TrendingUp className="w-5 h-5 text-indigo-400" />
          <h3 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>Live Metrics</h3>
        </div>
        <div className="h-[300px] flex items-center justify-center">
          <div className="skeleton w-full h-full" />
        </div>
      </div>
    );
  }

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass-card-static p-4" style={{ borderColor: 'var(--border-subtle)' }}>
          <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>{label}</p>
          <div className="space-y-1">
            <p className="text-sm">
              <span style={{ color: '#10b981' }}>Accuracy:</span>{" "}
              <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                {(payload[0].value * 100).toFixed(1)}%
              </span>
            </p>
            <p className="text-sm">
              <span style={{ color: '#6366f1' }}>Fairness (DPR):</span>{" "}
              <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                {payload[1].value.toFixed(3)}
              </span>
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: '0.3s' }}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-indigo-400" />
          <h3 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>Accuracy vs Fairness (Live)</h3>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#10b981' }} />
            <span style={{ color: 'var(--text-secondary)' }}>Accuracy</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#6366f1' }} />
            <span style={{ color: 'var(--text-secondary)' }}>Fairness (DPR)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-0.5" style={{ backgroundColor: 'rgba(239, 68, 68, 0.5)', borderTop: '2px dashed #ef4444' }} />
            <span style={{ color: 'var(--text-secondary)' }}>Threshold</span>
          </div>
        </div>
      </div>

      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="fairnessGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
            <XAxis 
              dataKey="time" 
              stroke="var(--text-muted)"
              tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
              axisLine={{ stroke: 'var(--border-subtle)' }}
            />
            <YAxis 
              stroke="var(--text-muted)"
              tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
              axisLine={{ stroke: 'var(--border-subtle)' }}
              domain={[0, 1.2]}
              ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine 
              y={fairnessThreshold} 
              stroke="#ef4444" 
              strokeDasharray="5 5"
              strokeOpacity={0.7}
              label={{ 
                value: 'Fair Threshold (0.8)', 
                position: 'insideTopRight',
                fill: '#ef4444',
                fontSize: 12 
              }}
            />
            <Area
              type="monotone"
              dataKey="accuracy"
              stroke="#10b981"
              strokeWidth={2}
              fill="url(#accuracyGradient)"
              animationDuration={500}
            />
            <Area
              type="monotone"
              dataKey="fairness"
              stroke="#6366f1"
              strokeWidth={2}
              fill="url(#fairnessGradient)"
              animationDuration={500}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
