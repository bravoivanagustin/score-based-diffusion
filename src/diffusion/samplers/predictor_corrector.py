"""Sampler predictor–corrector (Song et al., ICLR 2021).

Alterna un paso de Euler–Maruyama de la SDE reversa (predictor) con correcciones de
Langevin al nuevo nivel de ruido ``t + dt`` (corrector), con tamaño de paso adaptativo
por target de SNR.
"""

from __future__ import annotations

import math

import torch

from diffusion.sde import ForwardSDE

from .base import ReverseSampler, ScoreFn
from .time_grid import TimeGridFn

#: Piso de ``‖s‖`` en la fórmula de ``ε``: evita división por cero cuando el score es ~0.
_NORM_EPS: float = 1e-8


class PredictorCorrector(ReverseSampler):
    """Sampler predictor–corrector — estocástico.

    Combina, en cada paso, un predictor de Euler–Maruyama de la SDE reversa

        ``x ← x + (f - g² s)·dt + g·√|dt|·Z``

    con ``n_corrector`` pasos correctores de Langevin al nuevo nivel de ruido
    ``t + dt``

        ``x ← x + ε·s + √(2ε)·Z``,  con ``ε = 2·(snr·‖Z‖ / ‖s‖)²``

    (normas L2 medias por batch).
    """

    name = "pc"

    def __init__(
        self,
        sde: ForwardSDE,
        score_fn: ScoreFn,
        *,
        n_steps: int = 500,
        t_eps: float = 1e-3,
        time_grid: str | TimeGridFn = "uniform",
        n_corrector: int = 1,
        snr: float = 0.16,
    ) -> None:
        """Inicializa el sampler predictor–corrector.

        Args:
            sde: Proceso forward del que se derivan ``(f, g)`` y el prior ``p_T``.
            score_fn: Función pura ``(x, t) -> score`` que aproxima ``∇_x log p_t(x)``.
            n_steps: Número de pasos (intervalos) de integración; ``>= 1``.
            t_eps: Tiempo terminal; ``0 < t_eps < sde.T``.
            time_grid: Grilla temporal; ver :mod:`diffusion.samplers.time_grid`.
            n_corrector: Número de correcciones de Langevin por paso (``>= 0``); con ``0`` el
                paso se reduce al predictor de Euler–Maruyama.
            snr: Target de SNR que fija el tamaño de paso ``ε`` del corrector.

        Raises:
            ValueError: Si ``n_steps < 1``, ``t_eps`` cae fuera de ``(0, sde.T)`` o
                ``n_corrector < 0``.
        """
        super().__init__(sde, score_fn, n_steps=n_steps, t_eps=t_eps, time_grid=time_grid)
        if n_corrector < 0:
            raise ValueError(f"n_corrector debe ser >= 0; recibí n_corrector={n_corrector}")
        self.n_corrector = int(n_corrector)
        self.snr = float(snr)

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Avanza un paso predictor–corrector.

        Predictor: un paso de Euler–Maruyama de la SDE reversa de ``t`` a ``t + dt``.
        Corrector: ``n_corrector`` pasos de Langevin al nivel de ruido ``t + dt``.

        Args:
            x: Estado actual de shape ``(B, data_dim)``.
            t: Tiempo actual de shape ``(B,)`` o ``(B, 1)``.
            dt: Tamaño de paso (negativo: tiempo decreciente).
            generator: Generador del que se sortea el ruido.

        Returns:
            El nuevo estado de shape ``(B, data_dim)``.
        """
        t = self._expand_t(t)

        # --- Predictor: Euler–Maruyama de la SDE reversa.
        drift = self._reverse_drift(x, t)
        _, g = self.sde.sde(x, t)
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        x = x + drift * dt + g * math.sqrt(abs(dt)) * noise

        # --- Corrector: n_corrector pasos de Langevin al nuevo nivel de ruido t + dt.
        t_next = t + dt
        for _ in range(self.n_corrector):
            grad = self.score_fn(x, t_next)
            z = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            grad_norm = grad.reshape(grad.shape[0], -1).norm(dim=1).mean()
            noise_norm = z.reshape(z.shape[0], -1).norm(dim=1).mean()
            # ε adaptativo por target de SNR; piso en ‖s‖ contra div-by-zero / NaN.
            eps = 2.0 * (self.snr * noise_norm / grad_norm.clamp_min(_NORM_EPS)) ** 2
            x = x + eps * grad + torch.sqrt(2.0 * eps) * z

        return x
