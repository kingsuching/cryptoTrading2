"""SVM (Support Vector Machine) regression model for crypto price prediction."""
import numpy as np
import pandas as pd
from sklearn.svm import SVR


class SVMModel:
    """Wrapper around sklearn SVR for crypto price prediction."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: str = "scale",
    ):
        self.params = dict(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.model = SVR(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> dict:
        return self.params.copy()
