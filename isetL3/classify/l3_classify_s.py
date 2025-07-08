from abc import ABC, abstractmethod
from typing import Any, Tuple


class L3ClassifyS(ABC):
    """Abstract base class for L3 data classification."""

    def __init__(
        self,
        name: str = "",
        patch_size: int = 0,
        p_max: int = 0,
    ) -> None:
        self.name = name
        self.patch_size = patch_size
        self.p_max = p_max

    @abstractmethod
    def classify(self, *args: Any, **kwargs: Any) -> Any:
        """Return labels for input data."""
        raise NotImplementedError

    @abstractmethod
    def clear_data(self, *args: Any, **kwargs: Any) -> None:
        """Clear intermediate statistics and computed labels."""
        raise NotImplementedError

    @abstractmethod
    def get_class_data(
        self,
        label: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Return patches for a class and their locations."""
        raise NotImplementedError
