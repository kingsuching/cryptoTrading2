"""ARIMA model wrapper for univariate crypto price time-series forecasting."""
import numpy as np
import pandas as pd


class ARIMAModel:
    """Wrapper around statsmodels ARIMA for crypto price prediction.

    Parameters
    ----------
    p : int  AR order (auto-regression lag)
    d : int  Integration order (differencing)
    q : int  MA order (moving average)
    """

    def __init__(self, p: int = 1, d: int = 1, q: int = 1):
        self.p = p
        self.d = d
        self.q = q
        self.order = (p, d, q)
        self._result = None

    def fit(self, series: pd.Series) -> "ARIMAModel":
        """Fit on a univariate price series."""
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(series, order=self.order)
        self._result = model.fit()
        return self

    def forecast(self, steps: int = 7) -> np.ndarray:
        """Return `steps` out-of-sample point forecasts."""
        if self._result is None:
            raise RuntimeError("Model not fitted.")
        return self._result.forecast(steps=steps).values

    def predict_in_sample(self) -> np.ndarray:
        """Return in-sample fitted values."""
        if self._result is None:
            raise RuntimeError("Model not fitted.")
        return self._result.fittedvalues.values

    @property
    def aic(self) -> float:
        return self._result.aic if self._result is not None else float("inf")

    @property
    def bic(self) -> float:
        return self._result.bic if self._result is not None else float("inf")

    def summary(self):
        if self._result is not None:
            return self._result.summary()

    def get_order(self) -> tuple:
        return self.order
