from typing import List, Any, Tuple

from .l3_data_s import L3DataS


class L3DataSimulation(L3DataS):
    """Placeholder for ISET camera simulation data."""

    def __init__(
        self,
        camera: Any = None,
        sources: List[Any] | None = None,
        name: str = "default",
    ) -> None:
        super().__init__(name)
        self.camera = camera
        self.sources = sources or []
        self.exp_frac = [1, 0.6, 0.3, 0.1]
        self.ideal_cmf: Any = None
        self.verbose = True

    @property
    def cfa(self) -> Any:
        if self.camera is not None:
            return getattr(self.camera, "cfa", None)
        return None

    def data_get(
        self, n_img: int = 1, *args: Any, **kwargs: Any
    ) -> Tuple[List[Any], List[Any], Any]:
        # Placeholder: this should run an image simulation
        return [], [], self.cfa
