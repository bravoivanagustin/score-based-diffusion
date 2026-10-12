"""Clase base de los procesos forward ``dx = f(x, t) dt + g(t) dW``.

Cada SDE genera los pares de entrenamiento ``x_t`` y el target del score
:math:`\\nabla_x \\log p_t(x_t \\mid x_0)`. Cambiar la SDE cambia ``p_t`` y exige
reentrenar la red; cambiar el sampler reusa el mismo score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ForwardSDE(ABC):
    """Base de los procesos forward.

    Las subclases fijan :attr:`name` e implementan :meth:`sde`, :meth:`marginal_prob` y
    :meth:`prior_sampling`. :meth:`perturb` y :meth:`score_target` asumen un kernel
    ``N(mean, std^2 I)`` con ``std`` escalar por muestra y se derivan de :meth:`marginal_prob`.
    """

    #: Clave usada en el registry y la factory. Sobreescribir en cada subclase.
    name: str = ""

    #: Piso para ``std`` antes de dividir (evita división por cero en ``t -> 0``).
    _std_eps: float = 1e-5

    def __init__(self, data_dim: int | tuple[int, ...] = 2, T: float = 1.0) -> None:
        """Inicializa la SDE.

        Args:
            data_dim: Geometría del dato: entero ``d`` (forma ``(d,)``) o tupla
                ``(C, H, W)``. Se guarda crudo en :attr:`data_dim` y normalizado en
                :attr:`data_shape`.
            T: Horizonte temporal. El proceso corre en ``t in [0, T]``.

        Raises:
            ValueError: Si la forma es inválida (entero < 1, tupla vacía o alguna
                dimensión < 1).
        """
        #: Forma de evento normalizada ``(d,)`` / ``(C, H, W)``.
        self.data_shape: tuple[int, ...] = self._normalize_shape(data_dim)
        #: Valor crudo tal cual se pasó (se persiste en la metadata del checkpoint).
        self.data_dim: int | tuple[int, ...] = data_dim
        self.T = float(T)

    def hparams(self) -> dict:
        """Hiperparámetros del constructor (sin ``data_dim``), para persistir en el checkpoint.

        Cumple ``make_sde(sde.name, data_dim=sde.data_dim, **sde.hparams())`` ≡ ``sde``. Las
        subclases con parámetros propios extienden este dict.
        """
        return {"T": self.T}

    @staticmethod
    def _normalize_shape(data_dim: int | tuple[int, ...]) -> tuple[int, ...]:
        """Normaliza ``data_dim`` a una tupla de forma de evento y la valida.

        Raises:
            ValueError: Si es un entero < 1, una tupla vacía o con alguna dimensión < 1.
        """
        shape = (data_dim,) if isinstance(data_dim, int) else tuple(data_dim)
        if len(shape) == 0 or any(d < 1 for d in shape):
            raise ValueError(
                "la forma del dato debe ser un entero >= 1 o una tupla no vacía con toda "
                f"dimensión >= 1; recibí data_dim={data_dim!r}"
            )
        return shape

    # ------------------------------------------------------------- a implementar

    @abstractmethod
    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Coeficientes ``(drift, diffusion)`` de ``dx = f(x,t) dt + g(t) dW``.

        Args:
            x: Estado de shape ``(B, *E)`` para cualquier forma de evento ``E``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            ``(drift, diffusion)`` con ``drift`` de shape ``(B, *E)`` y ``diffusion`` con
            shape ``(B, 1, …, 1)`` (broadcastea sobre las dimensiones de evento).
        """
        raise NotImplementedError

    @abstractmethod
    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Media y desvío del kernel de perturbación ``p_t(x_t | x_0) = N(mean, std^2 I)``.

        Args:
            x0: Dato limpio de shape ``(B, *E)`` para cualquier forma de evento ``E``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            ``(mean, std)`` con ``mean`` de shape ``(B, *E)`` y ``std`` de shape
            ``(B, 1, …, 1)`` (se broadcastea sobre las dimensiones de evento).
        """
        raise NotImplementedError

    @abstractmethod
    def prior_sampling(
        self,
        shape: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Muestrea de la distribución terminal ``p_T`` (el prior del sampler).

        Args:
            shape: Shape de la salida, p. ej. ``(B, data_dim)``.
            generator: Generador de torch opcional para reproducibilidad.
            device: Dispositivo de la salida.
            dtype: Tipo de la salida (default ``float32``).

        Returns:
            Tensor de shape ``shape`` muestreado de ``p_T``.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------- API

    def perturb(
        self, x0: torch.Tensor, t: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Muestrea ``x_t`` del kernel de perturbación y devuelve el ruido usado.

        ``x_t = mean + std * eps`` con ``eps ~ N(0, I)``.

        Args:
            x0: Dato limpio de shape ``(B, *E)`` para cualquier forma de evento ``E``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.
            generator: Generador opcional para reproducibilidad.

        Returns:
            ``(x_t, eps)``, ambos de shape ``(B, *E)``.
        """
        mean, std = self.marginal_prob(x0, t)
        eps = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
        x_t = mean + std * eps
        return x_t, eps

    def score_target(
        self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score real del kernel y peso de la pérdida para denoising score matching.

        Con ``x_t = mean + std * eps``::

            ∇_{x_t} log p_t(x_t | x_0) = -(x_t - mean) / std^2 = -eps / std

        y peso ``lambda(t) = std^2``, que vuelve la pérdida ``|| std * s_theta + eps ||^2``.

        Args:
            x0: Dato limpio de shape ``(B, *E)`` para cualquier forma de evento ``E``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.
            eps: Ruido usado en :meth:`perturb`, shape ``(B, *E)``.

        Returns:
            ``(score_real, weight)`` con ``score_real`` de shape ``(B, *E)`` y ``weight``
            (peso por muestra) de shape ``(B, 1, …, 1)``.
        """
        _, std = self.marginal_prob(x0, t)
        std = std.clamp_min(self._std_eps)
        score_real = -eps / std
        weight = std**2
        return score_real, weight

    # ----------------------------------------------------------------- internos

    @staticmethod
    def _expand_t(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Reshapea ``t`` a ``(B, 1, …, 1)`` para que broadcastee contra ``ref`` ``(B, *E)``."""
        return t.reshape(t.shape[0], *([1] * (ref.ndim - 1)))

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return f"{type(self).__name__}(T={self.T})"
