import math
from abc import ABC, abstractmethod

import torch


@torch.jit.script
def _quintic_edge_jit(
    signed_dist: torch.Tensor, inv_overlap: torch.Tensor
) -> torch.Tensor:
    t = torch.clamp(signed_dist * inv_overlap + 0.5, 0.0, 1.0)
    return t * t * t * (10.0 + t * (6.0 * t - 15.0))


def prod_last_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Product over the last dim via explicit unrolled multiply, instead of
    `torch.prod`.

    Mathematically identical to `torch.prod(x, dim=-1, keepdim=True)` --
    not an approximation, bit-exact. But `torch.prod`'s backward has a
    data-dependent branch: it checks for exact 0.0 elements (a
    device->host sync) and, if any are found, falls back to a two-sided
    cumprod algorithm to avoid dividing by zero. Finite-support windows
    (quintic/cubic/raised_cosine) hit exact 0.0 outside their support on
    essentially every call, so they pay that sync+fallback every time
    `torch.prod` is used here. Elementwise `*` has no such branch
    (its backward never divides), so unrolling sidesteps the problem entirely.
    `d` (spatial dims) is small (2-3) and known at call time, so this is cheap.
    """
    parts = torch.unbind(x, dim=-1)
    result = parts[0]
    for p in parts[1:]:
        result = result * p
    return result.unsqueeze(-1)


class WindowFunction(ABC):
    name: str = "base"

    def __init__(self, overlap: torch.Tensor, eps: float = 1e-4, *args, **kwargs):
        self.overlap = overlap  # (d, )
        self.eps = eps

    @abstractmethod
    def _edge(self, signed_dist: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def decay_distance(self) -> float:
        """
        Distance from a block's own boundary beyond which this window's
        edge is ~0.
        Used by FBPINN._compute_affected_radius to decide how many
        neighboring blocks a point's window can reach.
        """
        ...

    def __call__(
        self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """Per-dimension window contribution, before the product over dims."""
        res = self._edge(x - a) * self._edge(b - x)
        # return res
        return torch.clamp(res, min=1e-16)

    def get_config(self) -> dict:
        return {"type": self.name, "eps": self.eps}


class SigmoidWindow(WindowFunction):
    name = "sigmoid"

    def __init__(
        self, overlap, eps: float = 1e-4, omega: float = None, *args, **kwargs
    ):
        super().__init__(overlap, eps)
        self.omega = omega

    def _edge(self, signed_dist):
        d = torch.clamp(signed_dist * self.omega, -50.0, 50.0)
        return torch.clamp(torch.sigmoid(d), min=1e-16)

    def decay_distance(self) -> float:
        return math.log(1 / self.eps) / self.omega


class _FiniteSupportWindow(WindowFunction):
    """Shared clip-and-normalize logic"""

    def __init__(self, overlap, eps: float = 1e-4, *args, **kwargs):
        super().__init__(overlap, eps)
        self.half_overlap = overlap / 2.0  # (d, )
        self.inv_overlap = 1.0 / overlap  # (d, )

    def _t(self, signed_dist):
        # half_overlap = self.overlap / 2.0
        # return torch.clamp((signed_dist + half_overlap) / self.overlap, 0.0, 1.0)
        return torch.clamp(signed_dist * self.inv_overlap + 0.5, 0.0, 1.0)

    def decay_distance(self) -> float:
        ov = self.overlap
        ov = float(ov.max().item()) if torch.is_tensor(ov) else float(ov)
        return ov / 2.0


class QuinticWindow(_FiniteSupportWindow):
    name = "quintic"

    def _edge(self, signed_dist):
        # t = self._t(signed_dist)
        return _quintic_edge_jit(signed_dist, self.inv_overlap)


class CubicWindow(_FiniteSupportWindow):
    name = "cubic"

    def _edge(self, signed_dist):
        t = self._t(signed_dist)
        return 3 * t**2 - 2 * t**3


class RaisedCosineWindow(_FiniteSupportWindow):
    name = "raised_cosine"

    def _edge(self, signed_dist):
        t = self._t(signed_dist)
        return 0.5 * (1 - torch.cos(math.pi * t))


def get_window_fn(name: str) -> type(WindowFunction):
    classes = {
        "sigmoid": SigmoidWindow,
        "quintic": QuinticWindow,
        "cubic": CubicWindow,
        "raised_cosine": RaisedCosineWindow,
    }
    return classes[name]
