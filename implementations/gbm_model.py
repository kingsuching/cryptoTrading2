"""GBM (Gradient Boosting Machine) model using scikit-learn GradientBoostingRegressor."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class GBMModel:
    """Wrapper around sklearn GradientBoostingRegressor for crypto price prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self.model = GradientBoostingRegressor(**self.params)
        self.feature_names_: list = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBMModel":
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else []
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    def get_params(self) -> dict:
        return self.params.copy()
