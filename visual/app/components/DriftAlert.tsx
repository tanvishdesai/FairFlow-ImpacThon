"use client";

import { AlertTriangle, X } from "lucide-react";
import { useEffect, useState } from "react";

interface DriftAlertProps {
  isActive: boolean;
  samplesRemaining: number;
  onDismiss?: () => void;
}

export default function DriftAlert({ isActive, samplesRemaining, onDismiss }: DriftAlertProps) {
  const [visible, setVisible] = useState(false);
  const [animateIn, setAnimateIn] = useState(false);

  useEffect(() => {
    if (isActive) {
      setVisible(true);
      setTimeout(() => setAnimateIn(true), 50);
    } else {
      setAnimateIn(false);
      setTimeout(() => setVisible(false), 300);
    }
  }, [isActive]);

  if (!visible) return null;

  return (
    <div 
      className={`fixed top-4 left-1/2 transform -translate-x-1/2 z-50 transition-all duration-300 ${
        animateIn ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4'
      }`}
    >
      <div 
        className="flex items-center gap-4 px-6 py-4 rounded-xl shadow-2xl"
        style={{
          background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.9), rgba(220, 38, 38, 0.9))',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          backdropFilter: 'blur(10px)',
          boxShadow: '0 0 40px rgba(239, 68, 68, 0.5)',
        }}
      >
        <div className="animate-pulse">
          <AlertTriangle className="w-6 h-6 text-white" />
        </div>
        <div>
          <p className="text-white font-bold text-lg">⚠️ BIAS DRIFT ACTIVE</p>
          <p className="text-white/80 text-sm">
            Injecting 90% female applicants • {samplesRemaining} samples remaining
          </p>
        </div>
        <div className="ml-4 flex items-center gap-3">
          <div className="flex gap-1">
            {[...Array(3)].map((_, i) => (
              <div 
                key={i}
                className="w-2 h-6 rounded-full bg-white/50"
                style={{
                  animation: `pulse 1s ease-in-out ${i * 0.2}s infinite`,
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
