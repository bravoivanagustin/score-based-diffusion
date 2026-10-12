"""Sampler Probability-Flow ODE con paso de Euler.

Integra la ODE de flujo de probabilidad (Song et al., ICLR 2021), que comparte las
marginales ``p_t`` de la SDE reversa pero sin ruido:
``dx = [f(x,t) - ½ g(t)^2 ∇_x log p_t(x)] dt``.
"""

from __future__ import annotations

import torch

from .base import ReverseSampler


class ProbabilityFlowODE(ReverseSampler):
    """Sampler Probability-Flow ODE de paso Euler — determinístico.

    Discretiza la ODE de flujo de probabilidad ``dx = [f - ½ g^2 s] dt`` con un paso de
    Euler explícito. Con la grilla en tiempo decreciente (``dt < 0``):

        ``x ← x + (f - ½ g^2 s)·dt``.

    Sin ruido: ``generator`` se ignora.
    """

    name = "pf_ode"

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Avanza un paso de Euler de la ODE de flujo de probabilidad.

        Args:
            x: Estado actual de shape ``(B, data_dim)``.
            t: Tiempo actual de shape ``(B,)`` o ``(B, 1)``.
            dt: Tamaño de paso (negativo: tiempo decreciente).
            generator: Ignorado (sampler determinístico).

        Returns:
            El nuevo estado de shape ``(B, data_dim)``.
        """
        return x + self._pfode_drift(x, t) * dt
