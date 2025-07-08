from __future__ import annotations

from typing import Any, Tuple, List

import numpy as np

from .l3_classify_s import L3ClassifyS


class SimpleClassifier(L3ClassifyS):
    """Simplified classifier splitting patches by pixel type."""

    def __init__(self, patch_size: int = 5, p_max: int = 1000) -> None:
        super().__init__(name="simple", patch_size=patch_size, p_max=p_max)
        self.labels: List[np.ndarray] | None = None
        self.store_data = True
        self.p_data: List[np.ndarray] | None = None
        self.p_out: List[np.ndarray] | None = None

    @property
    def n_labels(self) -> int:
        if self.labels is None:
            return 0
        vals = [np.max(arr) for arr in self.labels]
        return int(np.max(vals) + 1)

    def classify(
        self, l3d: Any, *args: Any, **kwargs: Any
    ) -> List[np.ndarray]:
        raw_list, out_list, p_type = l3d.data_get(len(l3d.in_img))
        self.labels = []
        self.p_data = []
        self.p_out = []
        for raw, out in zip(raw_list, out_list):
            labels = p_type.copy()
            self.labels.append(labels)
            if self.store_data:
                self.p_data.append(raw.reshape(-1, 1))
                shaped = out.reshape(-1, out.shape[-1])
                self.p_out.append(shaped)
        return self.labels

    def clear_data(self, *args: Any, **kwargs: Any) -> None:
        self.p_data = None
        self.p_out = None
        self.labels = None

    def get_class_data(
        self, label: Any, *args: Any, **kwargs: Any
    ) -> Tuple[Any, Any]:
        if self.labels is None or self.p_data is None:
            return None, None
        all_data = []
        all_out = []
        for lbl, data, out in zip(self.labels, self.p_data, self.p_out):
            mask = lbl.reshape(-1) == label
            all_data.append(data[mask])
            all_out.append(out[mask])
        return np.vstack(all_data), np.vstack(all_out)
