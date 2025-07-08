from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from ..classify import L3ClassifyS


class L3TrainS(ABC):
    """Base class for L3 training."""

    def __init__(self, name: str = "", l3c: L3ClassifyS | None = None) -> None:
        self.name = name
        self.l3c = l3c
        self.kernels: List[Any] | None = None
        self.out_channel_names: List[str] | None = None

    @property
    def n_channel_out(self) -> int:
        if self.kernels:
            import numpy as np
            return np.asarray(self.kernels[0]).shape[1]
        elif self.l3c:
            return getattr(self.l3c, "n_channel_out", 0)
        return 0

    @abstractmethod
    def train(self, l3d: Any, *args: Any, **kwargs: Any) -> "L3TrainS":
        """Train kernels for each class."""

    def save(self, fname: str, keep_data: bool = False) -> None:
        import pickle
        if not keep_data and self.l3c is not None:
            self.l3c.clear_data()
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname: str) -> "L3TrainS":
        import pickle
        with open(fname, "rb") as f:
            return pickle.load(f)
