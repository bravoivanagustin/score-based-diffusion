"""Sampler Euler–Maruyama: discretización estocástica de la SDE reversa."""

from __future__ import annotations

import math

import torch

from .base import ReverseSampler


class EulerMaruyama(ReverseSampler):
    """Sampler Euler–Maruyama de la SDE reversa — estocástico.

    Discretiza ``dx = [f - g^2 s] dt + g dW̄`` con un paso de Euler–Maruyama. Con la grilla
    en tiempo decreciente (``dt < 0``), el incremento de ruido se escala por ``√|dt|``:

        ``x ← x + (f - g^2 s)·dt + g·√|dt|·Z``,  con ``Z ~ N(0, I)``.

    ``Z`` se sortea del ``generator`` (reproducible por semilla).
    """

    name = "euler"

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Avanza un paso de Euler–Maruyama de la SDE reversa.

        Args:
            x: Estado actual de shape ``(B, data_dim)``.
            t: Tiempo actual de shape ``(B,)`` o ``(B, 1)``.
            dt: Tamaño de paso (negativo: tiempo decreciente).
            generator: Generador del que se sortea el ruido.

        Returns:
            El nuevo estado de shape ``(B, data_dim)``.
        """
        drift = self._reverse_drift(x, t)
        _, g = self.sde.sde(x, self._expand_t(t))
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return x + drift * dt + g * math.sqrt(abs(dt)) * noise
