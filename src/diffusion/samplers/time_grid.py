"""Grillas temporales de integración del proceso reverso, de ``T`` a ``t_eps``.

Variantes registradas: ``uniform`` (espaciado constante en ``t``) y ``logsnr`` (espaciado
constante en ``λ(t) = log(α_t² / σ_t²)``, concentra pasos en ``t`` chico). También se
acepta un callable propio ``fn(n_steps, t_min, t_max)``. Cambiar la grilla no requiere
reentrenar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from diffusion.sde import ForwardSDE

#: Contrato del callable de grilla propia: ``(n_steps, t_min, t_max) -> (n_steps + 1,)``.
TimeGridFn = Callable[[int, float, float], torch.Tensor]

#: Iteraciones de bisección al invertir ``λ(t)``: 60 alcanzan la resolución de ``float32``.
_BISECTION_ITERS = 60

#: Piso de los logaritmos en ``λ = 2(log α_t − log σ_t)``: evita ``log(0)`` cuando ``σ_t → 0``.
_LOG_EPS = 1e-30


class TimeGrid(ABC):
    """Base abstracta de las grillas temporales de integración.

    ``grid(n_steps)`` devuelve un tensor ``float32`` de shape ``(n_steps + 1,)``,
    estrictamente decreciente, con extremos exactos ``T`` y ``t_eps``.

    Args:
        sde: Proceso forward (aporta ``T`` y los coeficientes ``(α_t, σ_t)``).
        t_eps: Tiempo terminal; debe cumplir ``0 < t_eps < sde.T``.

    Raises:
        ValueError: Si ``t_eps`` queda fuera de ``(0, sde.T)``.
    """

    #: Clave de la variante en el registry.
    name: str

    def __init__(self, sde: ForwardSDE, t_eps: float) -> None:
        if not 0.0 < t_eps < sde.T:
            raise ValueError(f"t_eps debe cumplir 0 < t_eps < sde.T={sde.T}; recibí t_eps={t_eps}")
        self.sde = sde
        self.T = float(sde.T)
        self.t_eps = float(t_eps)

    @abstractmethod
    def grid(self, n_steps: int) -> torch.Tensor:
        """Construye la grilla de ``n_steps + 1`` tiempos de ``T`` a ``t_eps``.

        Args:
            n_steps: Número de pasos (intervalos) de integración; ``>= 1``.

        Returns:
            Tensor ``float32`` de shape ``(n_steps + 1,)``, decreciente, con extremos ``T``
            y ``t_eps``.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return f"{type(self).__name__}(T={self.T}, t_eps={self.t_eps})"


class UniformTimeGrid(TimeGrid):
    """Grilla uniforme en ``t``: ``torch.linspace(T, t_eps, n_steps + 1)``."""

    name = "uniform"

    def grid(self, n_steps: int) -> torch.Tensor:
        return torch.linspace(self.T, self.t_eps, n_steps + 1, dtype=torch.float32)


class LogSNRTimeGrid(TimeGrid):
    """Grilla uniforme en el log-SNR ``λ(t) = log(α_t² / σ_t²)``.

    Cada paso cambia la relación señal-ruido en la misma cantidad, lo que concentra pasos
    cerca de ``t → 0`` en VP/sub-VP (en VE coincide con ``uniform``). ``λ`` se obtiene de
    :meth:`~diffusion.sde.base.ForwardSDE.marginal_prob` y se invierte por bisección, así
    que funciona con cualquier SDE escalar-gaussiana con ``λ`` decreciente.
    """

    name = "logsnr"

    def _log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Evalúa ``λ(t) = 2(log α_t − log σ_t)`` para un tensor de tiempos ``(k,)``.

        ``marginal_prob`` se evalúa en ``x_0 = 1`` para que la media sea ``α_t``.

        Args:
            t: Tiempos de shape ``(k,)``.

        Returns:
            ``λ`` de shape ``(k,)`` en ``float32``, estrictamente decreciente en ``t``.
        """
        tt = t.reshape(-1, 1)
        alpha, sigma = self.sde.marginal_prob(torch.ones_like(tt), tt)
        log_snr = 2.0 * (
            torch.log(alpha.clamp_min(_LOG_EPS)) - torch.log(sigma.clamp_min(_LOG_EPS))
        )
        return log_snr.reshape(-1)

    def _invert_log_snr(self, targets: torch.Tensor) -> torch.Tensor:
        """Resuelve ``λ(t) = target`` por bisección, vectorizado sobre ``targets``.

        Args:
            targets: Valores de ``λ`` buscados, shape ``(k,)``; deben caer dentro de
                ``[λ(T), λ(t_eps)]``.

        Returns:
            Los tiempos ``t`` de shape ``(k,)`` en ``float32``.
        """
        lo = torch.full_like(targets, self.t_eps)  # λ(lo) = máximo
        hi = torch.full_like(targets, self.T)  # λ(hi) = mínimo
        for _ in range(_BISECTION_ITERS):
            mid = 0.5 * (lo + hi)
            # λ decrece en t: si λ(mid) sigue por encima del target, la raíz está a la derecha.
            go_right = self._log_snr(mid) > targets
            lo = torch.where(go_right, mid, lo)
            hi = torch.where(go_right, hi, mid)
        return 0.5 * (lo + hi)

    def grid(self, n_steps: int) -> torch.Tensor:
        bounds = self._log_snr(torch.tensor([self.T, self.t_eps], dtype=torch.float32))
        targets = torch.linspace(
            float(bounds[0]), float(bounds[1]), n_steps + 1, dtype=torch.float32
        )
        t = self._invert_log_snr(targets)
        # Extremos exactos en vez del residuo de la bisección.
        t[0] = self.T
        t[-1] = self.t_eps
        return t


class CallableTimeGrid(TimeGrid):
    """Adaptador de un callable propio a :class:`TimeGrid`, con validación.

    La salida puede ser cualquier cosa convertible a tensor; se valida antes de integrar.

    Args:
        fn: Callable ``(n_steps, t_min, t_max) -> (n_steps + 1,)``, decreciente de ``t_max``
            a ``t_min``. Se lo llama como ``fn(n_steps, t_eps, T)``.
        sde: Proceso forward (aporta ``T``).
        t_eps: Tiempo terminal; ``0 < t_eps < sde.T``.
    """

    name = "callable"

    def __init__(self, fn: TimeGridFn, sde: ForwardSDE, t_eps: float) -> None:
        super().__init__(sde, t_eps)
        if not callable(fn):
            raise ValueError(
                "time_grid debe ser el nombre de una grilla registrada o un callable "
                f"(n_steps, t_min, t_max) -> tensor; recibí {type(fn).__name__}"
            )
        self.fn = fn

    def grid(self, n_steps: int) -> torch.Tensor:
        raw = self.fn(n_steps, self.t_eps, self.T)
        try:
            t = torch.as_tensor(raw, dtype=torch.float32).reshape(-1)
        except (TypeError, RuntimeError) as exc:  # pragma: no cover - defensivo
            raise ValueError(
                f"El callable de time_grid devolvió algo no convertible a tensor: {raw!r}"
            ) from exc
        _validate_grid(t, n_steps, self.T, self.t_eps)
        return t

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        name = getattr(self.fn, "__name__", repr(self.fn))
        return f"CallableTimeGrid({name}, T={self.T}, t_eps={self.t_eps})"


def _validate_grid(t: torch.Tensor, n_steps: int, T: float, t_eps: float) -> None:
    """Valida que ``t`` cumpla el contrato de grilla del driver.

    Args:
        t: Grilla candidata, ya como tensor ``float32`` 1-D.
        n_steps: Número de pasos esperado (la grilla debe tener ``n_steps + 1`` puntos).
        T: Extremo inicial esperado.
        t_eps: Extremo final esperado.

    Raises:
        ValueError: Si el largo no es ``n_steps + 1``, hay valores no finitos, la grilla
            no es estrictamente decreciente o los extremos no coinciden con ``(T, t_eps)``.
    """
    expected = n_steps + 1
    if t.numel() != expected:
        raise ValueError(
            f"La grilla temporal debe tener n_steps + 1 = {expected} puntos; recibí {t.numel()}"
        )
    if not bool(torch.isfinite(t).all()):
        raise ValueError("La grilla temporal contiene valores no finitos (NaN/inf).")
    if expected > 1 and not bool((t[1:] < t[:-1]).all()):
        raise ValueError(
            "La grilla temporal debe ser estrictamente decreciente (se integra de T a "
            "t_eps); recibí una grilla no monótona o con puntos repetidos."
        )
    tol = 1e-5 + 1e-3 * max(abs(T), abs(t_eps))
    if abs(float(t[0]) - T) > tol or abs(float(t[-1]) - t_eps) > tol:
        raise ValueError(
            f"La grilla temporal debe ir de T={T} a t_eps={t_eps} (el prior se sortea en T "
            f"y el denoising final ocurre en t_eps); recibí extremos "
            f"({float(t[0])}, {float(t[-1])})."
        )


REGISTRY: dict[str, type[TimeGrid]] = {cls.name: cls for cls in (UniformTimeGrid, LogSNRTimeGrid)}


def available_time_grids() -> list[str]:
    """Nombres de las grillas temporales registradas, ordenados."""
    return sorted(REGISTRY)


def make_time_grid(spec: str | TimeGridFn, sde: ForwardSDE, t_eps: float) -> TimeGrid:
    """Crea la grilla temporal ``spec`` para ``sde``.

    Args:
        spec: Nombre registrado (``"uniform"``, ``"logsnr"``) o un callable
            ``(n_steps, t_min, t_max) -> (n_steps + 1,)`` decreciente.
        sde: Proceso forward del que salen ``T`` y los coeficientes del kernel.
        t_eps: Tiempo terminal; debe cumplir ``0 < t_eps < sde.T``.

    Raises:
        ValueError: Si ``spec`` es un string no registrado, no es ni string ni callable, o
            ``t_eps`` cae fuera de ``(0, sde.T)``.
    """
    if isinstance(spec, str):
        try:
            cls = REGISTRY[spec]
        except KeyError:
            opts = ", ".join(available_time_grids())
            raise ValueError(
                f"Grilla temporal desconocida '{spec}'. Opciones: {opts} "
                "(o pasá un callable (n_steps, t_min, t_max) -> tensor)."
            ) from None
        return cls(sde, t_eps)
    return CallableTimeGrid(spec, sde, t_eps)
