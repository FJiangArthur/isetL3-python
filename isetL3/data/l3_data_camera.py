from typing import List, Any, Tuple

from .l3_data_s import L3DataS


class L3DataCamera(L3DataS):
    """Camera data stored in memory."""

    def __init__(
        self,
        in_img: List[Any],
        out_img: List[Any],
        cfa: Any,
        name: str = "l3 Camera Data Class Instance",
    ) -> None:
        super().__init__(name)
        self.in_img = list(in_img)
        self.out_img = list(out_img)
        self._cfa = cfa

    @property
    def cfa(self) -> Any:
        return self._cfa

    def data_get(
        self, n_img: int = 1, *args: Any, **kwargs: Any
    ) -> Tuple[List[Any], List[Any], Any]:
        return self.in_img[:n_img], self.out_img[:n_img], self.p_type
