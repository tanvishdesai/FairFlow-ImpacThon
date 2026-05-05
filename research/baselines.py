"""
Base models and non-RL baselines for the FairFlow paper.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - only relevant if xgboost is missing
    XGBClassifier = None

from research.metrics import build_stream_trace, compute_static_metrics, paper_selection_score, summarize_stream_trace


MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")


def _build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in frame.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _build_estimator(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1200, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    raise KeyError(f"Unknown model '{name}'")


def train_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_names: Iterable[str] = MODEL_NAMES,
    seed: int = 42,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Pipeline]:
    """Train the requested base models and optionally persist them."""
    models: Dict[str, Pipeline] = {}
    for name in model_names:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(X_train)),
                ("model", _build_estimator(name, seed)),
            ]
        )
        pipeline.fit(X_train, y_train.values.ravel())
        models[name] = pipeline

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, output_path / f"{name}.joblib")
    return models


def model_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores for any fitted sklearn-style pipeline."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
        if scores.ndim == 2:
            return scores[:, 1]
        return scores.astype(float)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        decision = np.asarray(decision, dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))
    preds = model.predict(X)
    return np.asarray(preds, dtype=float)


@dataclass
class GroupThresholdOptimizer:
    """Validation-tuned group-specific threshold baseline."""

    fairness_threshold: float = 0.8
    threshold_grid: tuple[float, ...] = tuple(np.round(np.linspace(0.15, 0.85, 15), 2))

    def fit(self, y_true: np.ndarray, scores: np.ndarray, protected: np.ndarray) -> "GroupThresholdOptimizer":
        best_score = -1e18
        best_thresholds = (0.5, 0.5)
        best_metrics: Dict[str, float] | None = None

        for threshold_privileged in self.threshold_grid:
            for threshold_unprivileged in self.threshold_grid:
                preds = self._predict_with_thresholds(
                    scores,
                    protected,
                    threshold_privileged,
                    threshold_unprivileged,
                )
                metrics = compute_static_metrics(
                    y_true,
                    preds,
                    protected,
                    scores=scores,
                    fairness_threshold=self.fairness_threshold,
                )
                objective = paper_selection_score(metrics, fairness_threshold=self.fairness_threshold)
                if objective > best_score:
                    best_score = objective
                    best_thresholds = (threshold_privileged, threshold_unprivileged)
                    best_metrics = metrics

        self.threshold_privileged = best_thresholds[0]
        self.threshold_unprivileged = best_thresholds[1]
        self.validation_metrics_ = best_metrics or {}
        return self

    @staticmethod
    def _predict_with_thresholds(
        scores: np.ndarray,
        protected: np.ndarray,
        threshold_privileged: float,
        threshold_unprivileged: float,
    ) -> np.ndarray:
        privileged_mask = protected == 1
        thresholds = np.where(privileged_mask, threshold_privileged, threshold_unprivileged)
        return (scores >= thresholds).astype(int)

    def predict(self, scores: np.ndarray, protected: np.ndarray) -> np.ndarray:
        return self._predict_with_thresholds(
            scores,
            protected,
            self.threshold_privileged,
            self.threshold_unprivileged,
        )


@dataclass
class RuleBasedController:
    """
    A simple online controller that intervenes when rolling fairness is low.
    """

    fairness_threshold: float = 0.8
    window_size: int = 50
    approve_cutoff_grid: tuple[float, ...] = tuple(np.round(np.linspace(0.30, 0.70, 9), 2))
    deny_cutoff_grid: tuple[float, ...] = tuple(np.round(np.linspace(0.20, 0.50, 7), 2))

    def fit(
        self,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
    ) -> "RuleBasedController":
        best_objective = -1e18
        best_params = (0.45, 0.30)
        best_metrics: Dict[str, float] | None = None

        for approve_cutoff in self.approve_cutoff_grid:
            for deny_cutoff in self.deny_cutoff_grid:
                decisions, interventions, _ = self.simulate(
                    y_true=y_true,
                    base_preds=base_preds,
                    base_scores=base_scores,
                    protected=protected,
                    approve_cutoff=approve_cutoff,
                    deny_cutoff=deny_cutoff,
                )
                metrics = compute_static_metrics(
                    y_true,
                    decisions,
                    protected,
                    scores=base_scores,
                    intervention_flags=interventions,
                    fairness_threshold=self.fairness_threshold,
                )
                objective = paper_selection_score(
                    metrics,
                    fairness_threshold=self.fairness_threshold,
                    intervention_weight=0.25,
                )
                if objective > best_objective:
                    best_objective = objective
                    best_params = (approve_cutoff, deny_cutoff)
                    best_metrics = metrics

        self.approve_cutoff = best_params[0]
        self.deny_cutoff = best_params[1]
        self.validation_metrics_ = best_metrics or {}
        return self

    def simulate(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
        approve_cutoff: Optional[float] = None,
        deny_cutoff: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        approve_cutoff = self.approve_cutoff if approve_cutoff is None else approve_cutoff
        deny_cutoff = self.deny_cutoff if deny_cutoff is None else deny_cutoff

        decisions = np.zeros_like(base_preds, dtype=int)
        interventions = np.zeros_like(base_preds, dtype=int)

        rolling_approved_privileged = deque()
        rolling_approved_unprivileged = deque()

        for idx, (base_pred, score, group) in enumerate(zip(base_preds, base_scores, protected)):
            privileged_rate = np.mean(rolling_approved_privileged) if rolling_approved_privileged else 0.5
            unprivileged_rate = np.mean(rolling_approved_unprivileged) if rolling_approved_unprivileged else 0.5
            current_dpr = (
                1.0 if privileged_rate == 0 and unprivileged_rate == 0
                else 0.0 if privileged_rate == 0
                else unprivileged_rate / privileged_rate
            )

            final_decision = int(base_pred)
            if current_dpr < self.fairness_threshold and group == 0 and base_pred == 0 and score >= approve_cutoff:
                final_decision = 1
            elif current_dpr > (1.0 / max(self.fairness_threshold, 1e-6)) and group == 1 and base_pred == 1 and score <= deny_cutoff:
                final_decision = 0

            decisions[idx] = final_decision
            interventions[idx] = int(final_decision != base_pred)

            target_queue = rolling_approved_privileged if group == 1 else rolling_approved_unprivileged
            target_queue.append(final_decision)
            if len(target_queue) > self.window_size:
                target_queue.popleft()

        trace = build_stream_trace(
            y_true=y_true,
            y_pred=decisions,
            protected=protected,
            window_size=self.window_size,
            fairness_threshold=self.fairness_threshold,
        )
        return decisions, interventions, trace


@dataclass
class OnlineGroupThresholdConfig:
    fairness_threshold: float = 0.8
    fairness_upper_target: float = 1.05
    window_size: int = 50
    warmup_grid: tuple[int, ...] = (10, 25)
    step_size_grid: tuple[float, ...] = (0.02, 0.05, 0.10)
    privileged_scale_grid: tuple[float, ...] = (0.0, 0.5, 1.0)
    min_threshold: float = 0.10
    max_threshold: float = 0.90


@dataclass
class OnlineGroupThresholdController:
    """
    A naive online threshold baseline that updates group-specific thresholds from
    the stream itself rather than fixing them upfront from an offline calibration step.
    """

    config: OnlineGroupThresholdConfig = field(default_factory=OnlineGroupThresholdConfig)

    def fit(
        self,
        *,
        y_true: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
    ) -> "OnlineGroupThresholdController":
        best_objective = -1e18
        best_params = (10, 0.05, 0.5)
        best_metrics: Dict[str, float] | None = None

        for warmup_steps in self.config.warmup_grid:
            for step_size in self.config.step_size_grid:
                for privileged_scale in self.config.privileged_scale_grid:
                    decisions, interventions, _ = self.simulate(
                        y_true=y_true,
                        base_scores=base_scores,
                        protected=protected,
                        warmup_steps=warmup_steps,
                        step_size=step_size,
                        privileged_scale=privileged_scale,
                    )
                    metrics = compute_static_metrics(
                        y_true,
                        decisions,
                        protected,
                        scores=base_scores,
                        intervention_flags=interventions,
                        fairness_threshold=self.config.fairness_threshold,
                    )
                    objective = paper_selection_score(
                        metrics,
                        fairness_threshold=self.config.fairness_threshold,
                        intervention_weight=0.20,
                    )
                    if objective > best_objective:
                        best_objective = objective
                        best_params = (warmup_steps, step_size, privileged_scale)
                        best_metrics = metrics

        self.warmup_steps = int(best_params[0])
        self.step_size = float(best_params[1])
        self.privileged_scale = float(best_params[2])
        self.validation_metrics_ = best_metrics or {}
        return self

    def simulate(
        self,
        *,
        y_true: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
        warmup_steps: Optional[int] = None,
        step_size: Optional[float] = None,
        privileged_scale: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        warmup_steps = self.warmup_steps if warmup_steps is None else warmup_steps
        step_size = self.step_size if step_size is None else step_size
        privileged_scale = self.privileged_scale if privileged_scale is None else privileged_scale

        threshold_privileged = 0.5
        threshold_unprivileged = 0.5
        decisions = np.zeros(len(base_scores), dtype=int)
        interventions = np.zeros(len(base_scores), dtype=int)

        rolling_approved_privileged = deque()
        rolling_approved_unprivileged = deque()
        threshold_privileged_history: list[float] = []
        threshold_unprivileged_history: list[float] = []

        base_preds = (np.asarray(base_scores, dtype=float) >= 0.5).astype(int)

        for idx, (score, group, base_pred) in enumerate(zip(base_scores, protected, base_preds)):
            score = float(score)
            group = int(group)
            threshold = threshold_privileged if group == 1 else threshold_unprivileged
            final_decision = int(score >= threshold)
            decisions[idx] = final_decision
            interventions[idx] = int(final_decision != int(base_pred))

            target_queue = rolling_approved_privileged if group == 1 else rolling_approved_unprivileged
            target_queue.append(final_decision)
            if len(target_queue) > self.config.window_size:
                target_queue.popleft()

            priv_rate = np.mean(rolling_approved_privileged) if rolling_approved_privileged else 0.5
            unpriv_rate = np.mean(rolling_approved_unprivileged) if rolling_approved_unprivileged else 0.5
            current_dpr = (
                1.0 if priv_rate == 0 and unpriv_rate == 0
                else 0.0 if priv_rate == 0
                else unpriv_rate / priv_rate
            )

            if idx >= warmup_steps and rolling_approved_privileged and rolling_approved_unprivileged:
                if current_dpr < self.config.fairness_threshold:
                    deficit = self.config.fairness_threshold - current_dpr
                    threshold_unprivileged = max(
                        self.config.min_threshold,
                        threshold_unprivileged - step_size * deficit,
                    )
                    threshold_privileged = min(
                        self.config.max_threshold,
                        threshold_privileged + privileged_scale * step_size * deficit,
                    )
                elif current_dpr > self.config.fairness_upper_target:
                    overshoot = current_dpr - self.config.fairness_upper_target
                    threshold_unprivileged = min(
                        self.config.max_threshold,
                        threshold_unprivileged + step_size * overshoot,
                    )
                    threshold_privileged = max(
                        self.config.min_threshold,
                        threshold_privileged - privileged_scale * step_size * overshoot,
                    )

            threshold_privileged_history.append(threshold_privileged)
            threshold_unprivileged_history.append(threshold_unprivileged)

        trace = build_stream_trace(
            y_true=y_true,
            y_pred=decisions,
            protected=protected,
            window_size=self.config.window_size,
            fairness_threshold=self.config.fairness_threshold,
        )
        trace["threshold_privileged"] = threshold_privileged_history
        trace["threshold_unprivileged"] = threshold_unprivileged_history
        return decisions, interventions, trace


def evaluate_static_method(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    scores: np.ndarray,
    intervention_flags: Optional[np.ndarray],
    fairness_threshold: float,
    window_size: int,
) -> dict:
    """Compute both static and rolling summaries for a method."""
    metrics = compute_static_metrics(
        y_true,
        y_pred,
        protected,
        scores=scores,
        intervention_flags=intervention_flags,
        fairness_threshold=fairness_threshold,
    )
    trace = build_stream_trace(
        y_true,
        y_pred,
        protected,
        window_size=window_size,
        fairness_threshold=fairness_threshold,
    )
    metrics.update(summarize_stream_trace(trace))
    return metrics
