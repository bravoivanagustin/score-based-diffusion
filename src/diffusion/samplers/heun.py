"""Sampler Heun: integración determinística de 2º orden de la PF-ODE.

Mismo flujo que :class:`~diffusion.samplers.pf_ode.ProbabilityFlowODE` pero con el
esquema de Heun (sampler por defecto de EDM, Karras et al. 2022): mejor precisión por NFE.
"""

from __future__ import annotations

import torch

from .base import ReverseSampler


class HeunODE(ReverseSampler):
    """Sampler Heun (ODE de flujo de probabilidad, 2º orden) — determinístico.

    Discretiza ``dx = [f - ½ g^2 s] dt`` con el método de Heun (predictor de Euler +
    corrección por regla del trapecio). Con ``d = _pfode_drift`` y la grilla en tiempo
    decreciente (``dt < 0``):

        - predictor:  ``x̂ = x + d(x, t)·dt``
        - corrector:  ``x ← x + ½·[d(x, t) + d(x̂, t + dt)]·dt``

    Cada paso cuesta dos evaluaciones del score. Sin ruido: ``generator`` se ignora.
    """

    name = "heun"

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Avanza un paso de Heun de la ODE de flujo de probabilidad.

        Args:
            x: Estado actual de shape ``(B, data_dim)``.
            t: Tiempo actual de shape ``(B,)`` o ``(B, 1)``.
            dt: Tamaño de paso (negativo: tiempo decreciente).
            generator: Ignorado (sampler determinístico).

        Returns:
            El nuevo estado de shape ``(B, data_dim)``.
        """
        t = self._expand_t(t)
        d1 = self._pfode_drift(x, t)
        x_pred = x + d1 * dt
        d2 = self._pfode_drift(x_pred, t + dt)
        return x + 0.5 * (d1 + d2) * dt
