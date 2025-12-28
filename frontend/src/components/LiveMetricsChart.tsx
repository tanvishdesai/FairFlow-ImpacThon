"use client";

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface DataPoint {
  timestamp: string;
  accuracy: number;
  dpr: number;
  interventionRate?: number;
}

interface LiveMetricsChartProps {
  data: DataPoint[];
  fairflowActive: boolean;
}

export default function LiveMetricsChart({ data, fairflowActive }: LiveMetricsChartProps) {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass-card p-4 shadow-xl">
          <p className="text-xs text-slate-400 mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              <span className="font-medium">{entry.name}:</span>{' '}
              {entry.value.toFixed(3)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-white">Live Performance Metrics</h2>
          <p className="text-sm text-slate-400 mt-1">
            Accuracy vs Demographic Parity Ratio over time
          </p>
        </div>
        <div className={`status-badge ${fairflowActive ? 'fair' : 'unfair'}`}>
          FairFlow: {fairflowActive ? 'Active' : 'Inactive'}
        </div>
      </div>

      <div className="chart-container h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="timestamp"
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
              tickFormatter={(value) => {
                const date = new Date(value);
                return `${date.getMinutes()}:${date.getSeconds().toString().padStart(2, '0')}`;
              }}
            />
            <YAxis
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
              domain={[0, 1]}
              tickFormatter={(value) => value.toFixed(1)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{
                paddingTop: '20px',
              }}
              formatter={(value) => (
                <span className="text-sm text-slate-300">{value}</span>
              )}
            />
            
            {/* Fair DPR Range Reference Lines */}
            <ReferenceLine
              y={0.8}
              stroke="#f59e0b"
              strokeDasharray="5 5"
              label={{
                value: 'Fair Threshold (0.8)',
                position: 'right',
                fill: '#f59e0b',
                fontSize: 10,
              }}
            />
            <ReferenceLine
              y={1.25}
              stroke="#f59e0b"
              strokeDasharray="5 5"
              label={{
                value: 'Fair Threshold (1.25)',
                position: 'right',
                fill: '#f59e0b',
                fontSize: 10,
              }}
            />

            <Line
              type="monotone"
              dataKey="accuracy"
              name="Accuracy"
              stroke="#10b981"
              strokeWidth={3}
              dot={false}
              activeDot={{ r: 6, fill: '#10b981', stroke: '#fff', strokeWidth: 2 }}
            />
            <Line
              type="monotone"
              dataKey="dpr"
              name="Demographic Parity Ratio"
              stroke="#3b82f6"
              strokeWidth={3}
              dot={false}
              activeDot={{ r: 6, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }}
            />
            {data[0]?.interventionRate !== undefined && (
              <Line
                type="monotone"
                dataKey="interventionRate"
                name="Intervention Rate"
                stroke="#a855f7"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Legend Explanation */}
      <div className="flex flex-wrap gap-6 mt-4 text-sm text-slate-400">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
          <span>Higher is better (Model Performance)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500"></div>
          <span>Closer to 1.0 is fairer (Bias Metric)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-amber-500"></div>
          <span>Legal fairness boundaries (0.8 - 1.25)</span>
        </div>
      </div>
    </div>
  );
}
