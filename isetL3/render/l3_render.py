from typing import Any


class L3Render:
    """Simple renderer applying learned kernels."""

    def __init__(self, name: str = "default") -> None:
        self.name = name

    def render(
        self,
        raw_data: Any,
        p_type: Any,
        l3t: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        from ..data import L3DataCamera

        l3c = l3t.l3c
        data_obj = L3DataCamera([raw_data], [raw_data], p_type)
        l3c.classify(data_obj, *args, **kwargs)
        labels = l3c.labels[0]
        kernels = l3t.kernels
        import numpy as np
        out = np.zeros_like(raw_data, dtype=float)
        for idx, kernel in enumerate(kernels):
            if kernel is None:
                continue
            mask = labels == idx
            X = raw_data[mask]
            X = np.expand_dims(X, axis=1)
            X = np.pad(X, ((0, 0), (1, 0)), constant_values=1)
            out[mask] = X @ kernel
        return out
