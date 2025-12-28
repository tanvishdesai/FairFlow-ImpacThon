"use client";

import { useState, useEffect, useCallback } from "react";

const API_BASE = "http://localhost:8000";

// ============================================
// Types
// ============================================

export interface Metrics {
    timestamp: string;
    accuracy: number;
    demographic_parity_ratio: number;
    demographic_parity_difference: number;
    privileged_approval_rate: number;
    unprivileged_approval_rate: number;
    is_fair: boolean;
    total_predictions: number;
    total_interventions: number;
    intervention_rate: number;
}

export interface AuditEntry {
    id: number;
    timestamp: string;
    base_prediction: number;
    final_decision: number;
    intervention_type: string;
    protected_value: number;
    true_label: number | null;
}

export interface SimulationStatus {
    is_running: boolean;
    samples_processed: number;
    current_accuracy: number;
    current_dpr: number;
    fairflow_active: boolean;
}

export interface Explanation {
    id: number;
    prediction: number;
    probability_approve: number;
    intervention_type: string;
    intervention_reason: string;
    detailed_reason: string;
    top_contributors: Array<{
        feature: string;
        value: number;
        contribution: number;
    }>;
    waterfall_plot: string;
}

export interface FairFlowStatus {
    active: boolean;
    rl_agent_loaded: boolean;
    mode: string;
}

// ============================================
// useMetrics Hook
// ============================================

export function useMetrics(pollInterval: number = 1000) {
    const [metrics, setMetrics] = useState<Metrics | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchMetrics = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/metrics`);
            if (!response.ok) throw new Error("Failed to fetch metrics");
            const data = await response.json();
            setMetrics(data);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchMetrics();
        const interval = setInterval(fetchMetrics, pollInterval);
        return () => clearInterval(interval);
    }, [fetchMetrics, pollInterval]);

    return { metrics, isLoading, error, refetch: fetchMetrics };
}

// ============================================
// useAuditLog Hook
// ============================================

export function useAuditLog(limit: number = 50, pollInterval: number = 1000) {
    const [entries, setEntries] = useState<AuditEntry[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchAuditLog = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/audit-log?limit=${limit}`);
            if (!response.ok) throw new Error("Failed to fetch audit log");
            const data = await response.json();
            setEntries(data);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
        } finally {
            setIsLoading(false);
        }
    }, [limit]);

    useEffect(() => {
        fetchAuditLog();
        const interval = setInterval(fetchAuditLog, pollInterval);
        return () => clearInterval(interval);
    }, [fetchAuditLog, pollInterval]);

    return { entries, isLoading, error, refetch: fetchAuditLog };
}

// ============================================
// useSimulation Hook
// ============================================

export function useSimulation(pollInterval: number = 1000) {
    const [status, setStatus] = useState<SimulationStatus | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchStatus = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/simulate/status`);
            if (!response.ok) throw new Error("Failed to fetch simulation status");
            const data = await response.json();
            setStatus(data);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
        } finally {
            setIsLoading(false);
        }
    }, []);

    const startSimulation = useCallback(async (speed: number = 1.0) => {
        try {
            const response = await fetch(
                `${API_BASE}/api/simulate/start?speed=${speed}`,
                { method: "POST" }
            );
            if (!response.ok) throw new Error("Failed to start simulation");
            await fetchStatus();
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
        }
    }, [fetchStatus]);

    const stopSimulation = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/simulate/stop`, {
                method: "POST",
            });
            if (!response.ok) throw new Error("Failed to stop simulation");
            await fetchStatus();
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
        }
    }, [fetchStatus]);

    const injectDrift = useCallback(async (ratio: number = 0.9, duration: number = 50) => {
        try {
            const response = await fetch(
                `${API_BASE}/api/simulate/inject-drift?unprivileged_ratio=${ratio}&duration=${duration}`,
                { method: "POST" }
            );
            if (!response.ok) throw new Error("Failed to inject drift");
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
        }
    }, []);

    useEffect(() => {
        fetchStatus();
        const interval = setInterval(fetchStatus, pollInterval);
        return () => clearInterval(interval);
    }, [fetchStatus, pollInterval]);

    return {
        status,
        isLoading,
        error,
        startSimulation,
        stopSimulation,
        injectDrift,
        refetch: fetchStatus,
    };
}

// ============================================
// useFairFlow Hook
// ============================================

export function useFairFlow() {
    const [status, setStatus] = useState<FairFlowStatus | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchStatus = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/fairflow/status`);
            if (!response.ok) throw new Error("Failed to fetch FairFlow status");
            const data = await response.json();
            setStatus(data);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
        } finally {
            setIsLoading(false);
        }
    }, []);

    const toggleFairFlow = useCallback(async (active: boolean) => {
        try {
            const response = await fetch(
                `${API_BASE}/api/fairflow/toggle?active=${active}`,
                { method: "POST" }
            );
            if (!response.ok) throw new Error("Failed to toggle FairFlow");
            await fetchStatus();
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
        }
    }, [fetchStatus]);

    useEffect(() => {
        fetchStatus();
    }, [fetchStatus]);

    return { status, isLoading, error, toggleFairFlow, refetch: fetchStatus };
}

// ============================================
// useExplanation Hook
// ============================================

export function useExplanation() {
    const [explanation, setExplanation] = useState<Explanation | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchExplanation = useCallback(async (predictionId: number) => {
        setIsLoading(true);
        try {
            const response = await fetch(
                `${API_BASE}/api/explain/${predictionId}`
            );
            if (!response.ok) throw new Error("Failed to fetch explanation");
            const data = await response.json();
            setExplanation(data);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
            setExplanation(null);
        } finally {
            setIsLoading(false);
        }
    }, []);

    const clearExplanation = useCallback(() => {
        setExplanation(null);
        setError(null);
    }, []);

    return {
        explanation,
        isLoading,
        error,
        fetchExplanation,
        clearExplanation,
    };
}

// ============================================
// useApiHealth Hook
// ============================================

export function useApiHealth(pollInterval: number = 5000) {
    const [isConnected, setIsConnected] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    const checkHealth = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/`);
            setIsConnected(response.ok);
        } catch {
            setIsConnected(false);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        checkHealth();
        const interval = setInterval(checkHealth, pollInterval);
        return () => clearInterval(interval);
    }, [checkHealth, pollInterval]);

    return { isConnected, isLoading };
}
