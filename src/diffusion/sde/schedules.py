"""Schedules de las SDEs: ``beta(t)`` lineal (VP / sub-VP) y ``sigma(t)`` geométrico (VE).

Operan sobre ``t`` ya expandido (ver :meth:`diffusion.sde.base.ForwardSDE._expand_t`) y
devuelven tensores de la misma shape.
"""

from __future__ import annotations

import torch


def linear_beta(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """Schedule lineal ``beta(t) = beta_min + t (beta_max - beta_min)``."""
    return beta_min + t * (beta_max - beta_min)


def linear_beta_integral(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """Integral cerrada ``∫_0^t beta(s) ds = beta_min t + ½ (beta_max - beta_min) t^2``."""
    return beta_min * t + 0.5 * (beta_max - beta_min) * t**2


def geometric_sigma(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Schedule geométrico ``sigma(t) = sigma_min (sigma_max / sigma_min)^t``."""
    return sigma_min * (sigma_max / sigma_min) ** t
