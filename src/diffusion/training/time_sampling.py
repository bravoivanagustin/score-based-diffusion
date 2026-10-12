"""Distribuciones de muestreo de ``t`` con corrección por importance sampling.

Con ``t`` uniforme, la franja de ``t`` chico (donde el score es más difícil) recibe muy pocas
muestras. Cada variante no uniforme devuelve además los pesos del likelihood ratio contra la
uniforme (``w(t) = p_unif(t)/q(t)``), así que la pérdida pesada conserva la esperanza: solo
reduce varianza. Variantes: ``uniform`` (default, ``weights=None``) y ``log_uniform``.

Uso típico::

    from diffusion.training import make_time_sampler

    sampler = make_time_sampler("log_uniform", T=sde.T, t_eps=1e-4)
    t, weights = sampler.sample(batch_size, generator=g)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

from .losses import sample_timesteps


class TimeSampler(ABC):
    """Base abstracta de las distribuciones de muestreo de ``t`` de entrenamiento.

    Args:
        T: Horizonte temporal (``sde.T``).
        t_eps: Piso del muestreo; debe cumplir ``0 < t_eps < T`` (evita ``t = 0``, donde el
            target del score ``-eps/σ_t`` diverge).

    Raises:
        ValueError: Si ``t_eps`` queda fuera del rango abierto ``(0, T)``.
    """

    #: Nombre con el que la variante se registra en la factory (lo define cada subclase).
    name: str

    def __init__(self, T: float, t_eps: float) -> None:
        if not 0.0 < t_eps < T:
            raise ValueError(
                f"t_eps debe estar en el rango abierto (0, T={T}); se recibió t_eps={t_eps}"
            )
        self.T = float(T)
        self.t_eps = float(t_eps)

    @abstractmethod
    def sample(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Muestrea ``n`` tiempos de entrenamiento con sus pesos de corrección.

        Args:
            n: Cantidad de tiempos (típicamente el tamaño del batch).
            generator: Generador opcional de torch para reproducibilidad.
            device: Dispositivo de la salida.

        Returns:
            Par ``(t, weights)``: ``t`` de shape ``(n,)`` float32 en ``[t_eps, T]``; ``weights``
            ``None`` (uniforme) o ``(n,)`` float32 positivo con el likelihood ratio contra la
            uniforme (``E_q[w] = 1``).
        """


class UniformTimeSampler(TimeSampler):
    """Muestreo uniforme en ``[t_eps, T]`` (default); los pesos son ``None``.

    Delega en :func:`~diffusion.training.losses.sample_timesteps`: con el mismo generator
    reproduce exactamente la misma secuencia de tiempos.
    """

    name = "uniform"

    def sample(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        t = sample_timesteps(n, self.T, self.t_eps, generator=generator, device=device)
        return t, None


class LogUniformTimeSampler(TimeSampler):
    """Muestreo log-uniforme en ``[t_eps, T]``: concentra muestras en ``t`` chico.

    Con ``u ~ U(0,1)`` y ``t = t_eps·(T/t_eps)^u``, ``log t`` queda uniforme en
    ``[log t_eps, log T]``: la densidad es ``q(t) = 1/(t·ln(T/t_eps))`` y cada década de ``t``
    recibe la misma masa (con ``t_eps=1e-4`` y ``T=1``, el 50% cae en ``[t_eps, 0.01]``).

    Pesos: ``w(t) = p_unif(t)/q(t) = t·ln(T/t_eps)/(T − t_eps)``, con ``E_q[w] = 1``.
    """

    name = "log_uniform"

    def sample(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        u = torch.rand(n, generator=generator, device=device, dtype=torch.float32)
        t = self.t_eps * (self.T / self.t_eps) ** u
        log_ratio = math.log(self.T / self.t_eps)
        weights = t * (log_ratio / (self.T - self.t_eps))
        return t, weights


REGISTRY: dict[str, type[TimeSampler]] = {
    cls.name: cls for cls in (UniformTimeSampler, LogUniformTimeSampler)
}


def available_time_samplers() -> list[str]:
    """Nombres de las distribuciones de muestreo disponibles, ordenados."""
    return sorted(REGISTRY)


def make_time_sampler(name: str, T: float, t_eps: float) -> TimeSampler:
    """Crea la distribución de muestreo de tiempos ``name``.

    Args:
        name: Nombre registrado (``"uniform"`` o ``"log_uniform"``).
        T: Horizonte temporal (``sde.T``).
        t_eps: Piso del muestreo; debe cumplir ``0 < t_eps < T``.

    Returns:
        La variante construida con ``(T, t_eps)``.

    Raises:
        ValueError: Si ``name`` no está registrado o si ``t_eps`` queda fuera de ``(0, T)``.
    """
    try:
        cls = REGISTRY[name]
    except KeyError:
        opts = ", ".join(available_time_samplers())
        raise ValueError(
            f"Distribución de muestreo desconocida '{name}'. Opciones: {opts}"
        ) from None
    return cls(T, t_eps)
