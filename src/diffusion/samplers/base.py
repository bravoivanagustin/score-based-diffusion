"""Base abstracta de los samplers del proceso reverso.

Un :class:`ReverseSampler` integra la SDE reversa de Anderson (1982)
``dx = [f(x,t) - g(t)^2 ∇_x log p_t(x)] dt + g(t) dW̄`` (o su PF-ODE) desde el prior
``p_T`` hasta ``t_eps``, reusando el score aprendido: cambiar de sampler no requiere
reentrenar. Cada subclase define solo :meth:`ReverseSampler.step`.
"""

from __future__ import annotations

import abc
from collections.abc import Callable

import torch

from diffusion.sde import ForwardSDE

from .time_grid import TimeGridFn, make_time_grid

#: Score como función pura: ``(x: (B, *E), t: (B,) | (B,1)) -> (B, *E)`` para cualquier
#: forma de evento ``E`` (una red entrenada o un score analítico).
ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ReverseSampler(abc.ABC):
    """Base de todos los samplers del proceso reverso.

    Las subclases fijan :attr:`name` e implementan :meth:`step`; el ABC aporta el driver
    :meth:`sample`, la grilla temporal y los drifts reversos compartidos.
    """

    #: Clave del registry/factory, p. ej. ``"euler"``. Sobreescribir en cada subclase.
    name: str = ""

    def __init__(
        self,
        sde: ForwardSDE,
        score_fn: ScoreFn,
        *,
        n_steps: int = 500,
        t_eps: float = 1e-3,
        time_grid: str | TimeGridFn = "uniform",
    ) -> None:
        """Inicializa el sampler.

        Args:
            sde: Proceso forward del que se derivan ``(f, g)`` y el prior ``p_T``.
            score_fn: Función pura ``(x, t) -> score`` que aproxima ``∇_x log p_t(x)``.
            n_steps: Número de pasos (intervalos) de integración; ``>= 1``.
            t_eps: Tiempo terminal (evita integrar hasta ``t = 0``); ``0 < t_eps < sde.T``.
            time_grid: Nombre registrado (``"uniform"``, ``"logsnr"``) o callable
                ``(n_steps, t_min, t_max) -> (n_steps + 1,)`` decreciente; ver
                :mod:`diffusion.samplers.time_grid`.

        Raises:
            ValueError: Si ``n_steps < 1``, si ``t_eps`` cae fuera de ``(0, sde.T)``, o si
                ``time_grid`` no es un nombre registrado ni un callable.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps debe ser >= 1; recibí n_steps={n_steps}")
        if not (0.0 < t_eps < sde.T):
            raise ValueError(f"t_eps debe cumplir 0 < t_eps < sde.T={sde.T}; recibí t_eps={t_eps}")
        self.sde = sde
        self.score_fn = score_fn
        self.n_steps = int(n_steps)
        self.t_eps = float(t_eps)
        self.time_grid = make_time_grid(time_grid, sde, self.t_eps)

    # --------------------------------------------------------------- a implementar

    @abc.abstractmethod
    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Avanza un paso de integración de ``t`` a ``t + dt`` (con ``dt < 0``).

        Args:
            x: Estado actual de shape ``(B, *E)``.
            t: Tiempo actual de shape ``(B,)`` o ``(B, 1)``.
            dt: Tamaño de paso (negativo: tiempo decreciente).
            generator: Generador de torch para los samplers estocásticos; los
                determinísticos (PF-ODE/Heun) lo ignoran.

        Returns:
            El nuevo estado de shape ``(B, *E)``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------- driver

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        *,
        init: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        return_trajectory: bool = False,
        denoise: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Integra el proceso reverso de ``T`` a ``t_eps`` y devuelve las muestras ``x_0``.

        Arranca del prior ``p_T`` (o de ``init``) y recorre la grilla en tiempo decreciente
        delegando cada paso en :meth:`step`, bajo ``torch.no_grad()`` y en ``float32``.

        Args:
            n_samples: Número de muestras a generar (``N``).
            init: Estado inicial ``x_T`` de shape ``(n_samples, *sde.data_shape)``; si es
                ``None`` se sortea de ``sde.prior_sampling``.
            generator: Generador para el prior y los pasos estocásticos. En GPU debe estar
                en el mismo device (p. ej. ``torch.Generator(device="cuda")``).
            device: Device donde sortear el prior cuando ``init`` es ``None`` (default CPU);
                el resto corre en el device de ``x``.
            return_trajectory: Si es ``True``, devuelve además la trayectoria completa.
            denoise: Si es ``True``, aplica un paso final de Tweedie en ``t_eps``
                (``E[x_0 | x_t] = (x + σ_t² s) / α_t``) en vez de devolver el estado crudo.

        Returns:
            ``x_0`` de shape ``(n_samples, *sde.data_shape)`` en ``float32``. Con
            ``return_trajectory``, la tupla ``(x_0, trayectoria)`` con trayectoria de shape
            ``(n_steps + 1, n_samples, *sde.data_shape)``, de ``x_T`` a ``x_0``.
        """
        if init is None:
            x = self.sde.prior_sampling(
                (n_samples, *self.sde.data_shape),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            x = init.to(dtype=torch.float32)

        # La grilla va al device de x para que t, la SDE y el score no mezclen CPU/GPU.
        grid = self._time_grid().to(x.device)
        trajectory: list[torch.Tensor] = [x.clone()] if return_trajectory else []

        for i in range(self.n_steps):
            t_cur = grid[i]
            t_next = grid[i + 1]
            dt = (t_next - t_cur).item()  # negativo: tiempo decreciente
            t_batch = t_cur.expand(n_samples, 1)
            x = self.step(x, t_batch, dt, generator=generator)
            if return_trajectory:
                trajectory.append(x.clone())

        # Denoising de Tweedie en t_eps; se mantiene el invariante trajectory[-1] == x_0.
        if denoise:
            x = self._denoise(x, grid[-1].expand(n_samples, 1))
            if return_trajectory:
                trajectory[-1] = x.clone()

        if return_trajectory:
            return x, torch.stack(trajectory, dim=0)
        return x

    # ----------------------------------------------------------- helpers compartidos

    def _time_grid(self) -> torch.Tensor:
        """Grilla temporal de ``T`` a ``t_eps``, según :attr:`time_grid`.

        Returns:
            Tensor ``float32`` de shape ``(n_steps + 1,)``, decreciente, con extremos
            ``T`` (primero) y ``t_eps`` (último).
        """
        return self.time_grid.grid(self.n_steps)

    def _reverse_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift de la SDE reversa ``f - g^2 s``.

        Args:
            x: Estado de shape ``(B, *E)`` para cualquier forma de evento ``E``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Tensor de shape ``(B, *E)``.
        """
        t = self._expand_t(t)
        f, g = self.sde.sde(x, t)
        s = self.score_fn(x, t)
        return f - (g**2) * s

    def _pfode_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift del flujo de probabilidad (PF-ODE) ``f - ½ g^2 s``.

        Comparte las mismas marginales que la SDE reversa pero sin término de ruido.

        Args:
            x: Estado de shape ``(B, *E)`` para cualquier forma de evento ``E``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Tensor de shape ``(B, *E)``.
        """
        t = self._expand_t(t)
        f, g = self.sde.sde(x, t)
        s = self.score_fn(x, t)
        return f - 0.5 * (g**2) * s

    def _denoise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Paso final de *denoising* de Tweedie: estima ``E[x_0 | x_t]`` en ``t_eps``.

        Para el kernel ``x_t = α_t x_0 + σ_t ε``: ``E[x_0 | x_t] = (x_t + σ_t² · s) / α_t``.
        ``(α_t, σ_t)`` salen de :meth:`ForwardSDE.marginal_prob` evaluada en ``x_0 = 1``;
        el denominador se acota por debajo por robustez.

        Args:
            x: Estado ``x_{t_eps}`` de shape ``(B, *E)``.
            t: Tiempo ``t_eps`` de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            La estimación limpia del dato ``x_0`` de shape ``(B, *E)``.
        """
        t = self._expand_t(t)
        mean_coef, std = self.sde.marginal_prob(torch.ones_like(x), t)  # α_t, σ_t
        s = self.score_fn(x, t)
        return (x + std**2 * s) / mean_coef.clamp_min(self.sde._std_eps)

    # ----------------------------------------------------------------- internos

    @staticmethod
    def _expand_t(t: torch.Tensor) -> torch.Tensor:
        """Normaliza ``t`` de shape ``(B,)`` o ``(B, 1)`` a ``(B, 1)`` para broadcast."""
        return t.reshape(-1, 1)

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"{type(self).__name__}(sde={type(self.sde).__name__}, "
            f"n_steps={self.n_steps}, t_eps={self.t_eps})"
        )
