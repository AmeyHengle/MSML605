from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class EvalResult:
    rmse: float
    mae: float
    r2: float
    mape: float

    def to_dict(self) -> dict[str, float]:
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2, "mape": self.mape}


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> EvalResult:
    """Compute regression metrics. Zero targets are excluded from MAPE."""
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(
        np.nanmean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    )
    return EvalResult(rmse=rmse, mae=mae, r2=r2, mape=mape)
