from abc import ABC, abstractmethod
from typing import List, Tuple, Any


class L3DataS(ABC):
    """Base class for L3 data generation."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.in_img: List[Any] = []
        self.out_img: List[Any] = []

    @property
    @abstractmethod
    def cfa(self) -> Any:
        """Return the CFA pattern."""

    @property
    def p_type(self) -> Any:
        if self.in_img:
            # Assume first image defines shape
            import numpy as np
            return np.zeros_like(self.in_img[0], dtype=int)
        return None

    @abstractmethod
    def data_get(
        self, n_img: int = 1, *args: Any, **kwargs: Any
    ) -> Tuple[List[Any], List[Any], Any]:
        """Return input and output image pairs."""
