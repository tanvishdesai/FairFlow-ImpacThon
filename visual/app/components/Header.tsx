"use client";

import { Shield, Activity, Zap, Sun, Moon } from "lucide-react";
import { useTheme } from "../context/ThemeContext";
import { useState, useEffect } from "react";

interface HeaderProps {
  isConnected: boolean;
  fairflowActive: boolean;
}

export default function Header({ isConnected, fairflowActive }: HeaderProps) {
  const { theme, toggleTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <header className="glass-card-static px-6 py-4 mb-6">
      <div className="flex items-center justify-between">
        {/* Logo & Title */}
        <div className="flex items-center gap-4">
          <div className="relative">
            <div className={`p-3 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 ${fairflowActive ? 'glow-blue' : ''}`}>
              <Shield className="w-8 h-8 text-white" />
            </div>
            {fairflowActive && (
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full animate-pulse-glow" style={{ borderWidth: '2px', borderStyle: 'solid', borderColor: 'var(--bg-primary)' }} />
            )}
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight" style={{ color: 'var(--text-primary)' }}>
              FairFlow
            </h1>
            <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              AI Governance Dashboard
            </p>
          </div>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-4">
          {/* Theme Toggle - only show icon after mount to prevent hydration mismatch */}
          <button
            onClick={toggleTheme}
            className="theme-toggle"
            aria-label="Toggle theme"
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {mounted ? (
              theme === 'dark' ? (
                <Sun className="w-5 h-5" style={{ color: '#facc15' }} />
              ) : (
                <Moon className="w-5 h-5" style={{ color: '#6366f1' }} />
              )
            ) : (
              <div className="w-5 h-5" />
            )}
          </button>

          {/* API Connection Status */}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              {isConnected ? 'API Connected' : 'API Disconnected'}
            </span>
          </div>

          {/* FairFlow Status */}
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${
            fairflowActive 
              ? 'bg-green-500/20 border border-green-500/30' 
              : 'border'
          }`} style={{ borderColor: fairflowActive ? undefined : 'var(--border-subtle)' }}>
            {fairflowActive ? (
              <Zap className="w-4 h-4 text-green-400" />
            ) : (
              <Activity className="w-4 h-4" style={{ color: 'var(--text-secondary)' }} />
            )}
            <span className="text-sm font-medium" style={{ color: fairflowActive ? '#10b981' : 'var(--text-secondary)' }}>
              {fairflowActive ? 'Protection Active' : 'Protection Off'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
