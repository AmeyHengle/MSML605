from __future__ import annotations

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.base import clone
from sklearn.linear_model import Ridge

from ml605_pipeline.evaluate import compute_metrics


# Five sklearn estimators to compare. Keys become MLflow run names.
CANDIDATE_MODELS: dict[str, object] = {
    "random_forest": RandomForestRegressor(
        n_estimators=300, max_depth=14, random_state=42, n_jobs=-1, oob_score=True
    ),
    "extra_trees": ExtraTreesRegressor(
        n_estimators=300, random_state=42, n_jobs=-1, bootstrap=True, oob_score=True
    ),
    "hist_gradient_boosting": HistGradientBoostingRegressor(
        max_iter=300, max_depth=10, random_state=42
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
    ),
    "ridge_baseline": Ridge(alpha=1.0),
}


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    model: object
    metrics: dict[str, float]
    rmse: float  # Primary selection criterion
    run_id: str = ""  # MLflow child run ID where this model's artifact is logged


@dataclass(frozen=True)
class AutoMLResult:
    best: ModelCandidate
    all_candidates: list[ModelCandidate]


def _evaluate_candidate(
    name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelCandidate:
    """Train one model and return its evaluation metrics. Does NOT start an MLflow run."""
    model.fit(X_train, y_train)  # type: ignore[union-attr]
    preds = model.predict(X_test)  # type: ignore[union-attr]
    preds_train = model.predict(X_train)  # type: ignore[union-attr]

    test_metrics = compute_metrics(y_test, preds)
    train_eval = compute_metrics(y_train, preds_train)

    metrics: dict[str, float] = {
        **test_metrics.to_dict(),
        "rmse_train": train_eval.rmse,
        "r2_train": train_eval.r2,
    }
    if hasattr(model, "oob_score_"):
        metrics["oob_score"] = float(model.oob_score_)  # type: ignore[union-attr]

    return ModelCandidate(name=name, model=model, metrics=metrics, rmse=test_metrics.rmse)


def run_automl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> AutoMLResult:
    """
    Train all CANDIDATE_MODELS, log each as a nested MLflow run.
    Returns AutoMLResult with the best model (lowest test RMSE).

    Must be called inside an active mlflow.start_run() context so nested runs attach.
    """
    candidates: list[ModelCandidate] = []

    for name, model_template in CANDIDATE_MODELS.items():
        model = clone(model_template)
        with mlflow.start_run(run_name=name, nested=True) as child_run:
            candidate = _evaluate_candidate(name, model, X_train, y_train, X_test, y_test)
            mlflow.log_param("model_type", name)
            mlflow.log_param("feature_count", X_train.shape[1])
            for k, v in candidate.metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(model, artifact_path="model")
            child_run_id = child_run.info.run_id
        # Rebuild with run_id (dataclass is frozen so we use replace pattern)
        candidate = ModelCandidate(
            name=candidate.name,
            model=candidate.model,
            metrics=candidate.metrics,
            rmse=candidate.rmse,
            run_id=child_run_id,
        )
        candidates.append(candidate)

    best = min(candidates, key=lambda c: c.rmse)
    return AutoMLResult(best=best, all_candidates=candidates)
