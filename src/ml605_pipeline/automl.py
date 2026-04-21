from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _train_candidates_sequential(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[ModelCandidate]:
    """Train all candidates one after another (legacy path, used for benchmarking)."""
    out: list[ModelCandidate] = []
    for name, template in CANDIDATE_MODELS.items():
        model = clone(template)
        out.append(_evaluate_candidate(name, model, X_train, y_train, X_test, y_test))
    return out


def _train_candidates_parallel(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_workers: int | None = None,
) -> list[ModelCandidate]:
    """
    Train all candidates concurrently using a thread pool.

    Threading is safe here because sklearn releases the GIL in the heavy numeric paths
    (BLAS/LAPACK for Ridge, tree-building inner loops for the ensembles), so the Python
    wall-clock gap shrinks to the slowest single candidate rather than the sum of all.
    MLflow is *not* called from worker threads — nested runs rely on thread-local state
    that does not propagate cleanly into the pool.
    """
    workers = max_workers or len(CANDIDATE_MODELS)
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for name, template in CANDIDATE_MODELS.items():
            model = clone(template)
            fut = executor.submit(
                _evaluate_candidate, name, model, X_train, y_train, X_test, y_test
            )
            futures[fut] = name

        completed: list[ModelCandidate] = []
        for fut in as_completed(futures):
            completed.append(fut.result())

    order = list(CANDIDATE_MODELS.keys())
    completed.sort(key=lambda c: order.index(c.name))
    return completed


def run_automl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    parallel: bool = True,
    max_workers: int | None = None,
) -> AutoMLResult:
    """
    Train all CANDIDATE_MODELS, log each as a nested MLflow run.
    Returns AutoMLResult with the best model (lowest test RMSE).

    Must be called inside an active mlflow.start_run() context so nested runs attach.

    Args:
        parallel: If True (default), candidates are trained concurrently in a thread
            pool and MLflow nested runs are logged sequentially afterward. If False,
            falls back to the original serial train-and-log loop.
        max_workers: Thread pool size when parallel=True. Defaults to the number
            of candidates so each model gets its own thread.
    """
    if parallel:
        trained = _train_candidates_parallel(
            X_train, y_train, X_test, y_test, max_workers=max_workers
        )
    else:
        trained = _train_candidates_sequential(X_train, y_train, X_test, y_test)

    candidates: list[ModelCandidate] = []
    for candidate in trained:
        with mlflow.start_run(run_name=candidate.name, nested=True) as child_run:
            mlflow.log_param("model_type", candidate.name)
            mlflow.log_param("feature_count", X_train.shape[1])
            for k, v in candidate.metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(candidate.model, artifact_path="model")
            child_run_id = child_run.info.run_id

        candidates.append(
            ModelCandidate(
                name=candidate.name,
                model=candidate.model,
                metrics=candidate.metrics,
                rmse=candidate.rmse,
                run_id=child_run_id,
            )
        )

    best = min(candidates, key=lambda c: c.rmse)
    return AutoMLResult(best=best, all_candidates=candidates)
