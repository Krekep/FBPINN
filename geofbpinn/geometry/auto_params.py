"""
Automatic calculation of FBPINN decomposition parameters.

Relationships between parameters:
- kappa = delta / B  (overlap / block_size) — overlap coefficient
- omega = 2 * ln(1 / eps) / (kappa * B) — sigmoid sharpness
- affected_radius — block influence radius (calculated in FBPINN, not here)

Usage:
    # Auto-calculate everything from block_size and eps:
    params = resolve_params(block_size=(0.2, 0.2), eps=1e-4)

    # Override kappa:
    params = resolve_params(block_size=(0.2, 0.2), eps=1e-4, kappa=0.3)

    # Override overlap (kappa will be derived):
    params = resolve_params(block_size=(0.2, 0.2), eps=1e-4, overlap=(0.06, 0.06))

    # Override omega manually:
    params = resolve_params(block_size=(0.2, 0.2), eps=1e-4, omega=50)
"""

import math
from typing import Optional, Tuple, Sequence


def resolve_overlap_and_kappa(
    overlap: Optional[Tuple[float, ...]],
    block_size: Tuple[float, ...],
    kappa: float = 0.3,
) -> Tuple[Tuple[float, ...], float]:
    """
    Resolve the overlap/kappa ambiguity. If overlap is not None, override kappa

    Returns
    -------
    (overlap, kappa) — final values
    """
    final_kappa = kappa
    if overlap is not None:
        final_overlap = tuple(overlap)
        final_kappa = overlap[0] / block_size[0]
    else:
        if kappa is None:
            final_kappa = 0.3
        final_overlap = tuple(final_kappa * b for b in block_size)

    return final_overlap, final_kappa


def compute_omega(
    eps: float,
    kappa: float,
    block_size: float,
) -> float:
    """
    Compute omega from eps, kappa and block_size.

    Formula: omega = 2 * ln(1 / eps) / (kappa * B)

    With tight omega, block weight drops to eps exactly at its own edge.
    Verification: x where (x - b) * omega = ln(1 / eps)

    Parameters
    ----------
    eps : float
        Numerical threshold (1e-4 .. 1e-6)
    kappa : float
        Overlap ratio (0.2 .. 0.47)
    block_size : float
        Block size along the axis used for calculation

    Returns
    -------
    float
        Computed omega value
    """
    assert (
        kappa > 0 and block_size > 0
    ), f"kappa={kappa}, block_size={block_size} must be positive"
    assert 0 < eps < 1, f"eps={eps} must be in (0, 1)"

    return 2.0 * math.log(1.0 / eps) / (kappa * block_size)


def resolve_params(
    block_size: Tuple[float, ...],
    eps: float = 1e-4,
    overlap: Optional[Tuple[float, ...]] = None,
    kappa: float = 0.3,
    omega: Optional[float] = None,
) -> tuple[Sequence[float], float, float]:
    """
    Compute the full set of decomposition parameters.

    Parameters
    ----------
    block_size : Tuple[float, ...]
        Block size per dimension
    eps : float
        Numerical threshold
    overlap : Optional[Tuple[float, float, ...]]
        Manual overlap override
    kappa : float
        Manual kappa override
    omega : Optional[float]
        Manual omega override
    default_kappa : float
        Default kappa if neither overlap nor kappa is given

    Returns
    -------
    overlap, kappa, omega
    """
    final_overlap, final_kappa = resolve_overlap_and_kappa(overlap, block_size, kappa)

    if omega is None:
        final_omega = compute_omega(eps, final_kappa, block_size[0])
    else:
        final_omega = omega

    return final_overlap, final_kappa, final_omega
