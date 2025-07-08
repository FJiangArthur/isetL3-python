from __future__ import annotations

from typing import Any

import numpy as np

from ..classify import L3ClassifyS
from .l3_train_s import L3TrainS


class L3TrainOLS(L3TrainS):
    """Ordinary least squares training implementation."""

    def __init__(
        self,
        l3c: L3ClassifyS | None = None,
        name: str = "l3 Train OLS instance",
        p_min: int = 50,
        verbose: bool = True,
    ) -> None:
        super().__init__(name=name, l3c=l3c)
        self.p_min = p_min
        self.verbose = verbose
        self.out_channel_names = ["red", "green", "blue"]

    def train(
        self, l3d: Any, *args: Any, **kwargs: Any
    ) -> "L3TrainOLS":
        l3c = self.l3c
        if l3c is None:
            raise ValueError("l3c must be set")
        l3c.classify(l3d, *args, **kwargs)
        n_labels = getattr(l3c, "n_labels", 0)
        self.kernels = [None] * n_labels
        for idx in range(n_labels):
            X, y = l3c.get_class_data(idx)
            if X is None or y is None or len(X) < self.p_min:
                continue
            X = np.pad(X, ((0, 0), (1, 0)), constant_values=1)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.kernels[idx] = beta
        return self
