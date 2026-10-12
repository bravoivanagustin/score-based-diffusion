"""Parametrización ε del score: ``score(x, t) = -inner(x, t) / clamp(σ_t, 1e-5)``.

A ``t`` chico el score real escala como ``1/σ_t``; así la red interna predice una cantidad
acotada (unidades de ε, como en DDPM / Song et al. 2021) y la magnitud sale del álgebra.
Con el peso ``λ(t) = σ²`` de la pérdida, la regresión interna queda en unidades de ε.

Uso típico::

    from diffusion.models import EpsilonScoreWrapper

    net = ScoreMLP(data_dim=2)
    wrapper = EpsilonScoreWrapper(net, lambda x, t: sde.marginal_prob(x, t)[1])
    score = wrapper(x, t)               # == -net(x, t) / clamp(σ_t, 1e-5)
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .base import ScoreModel

#: Piso para σ antes de dividir (igual a ``ForwardSDE._std_eps``).
_SIGMA_FLOOR: float = 1e-5


class EpsilonScoreWrapper(nn.Module):
    """Score = ``-inner(x, t) / clamp(σ(x, t), 1e-5)`` con el contrato ``(x, t) -> score``.

    Envuelve una red que predice en unidades de ε. No tiene parámetros propios.

    ``state_dict``/``load_state_dict`` delegan al interno sin el prefijo ``_net.``, así el
    checkpoint es de red pelada y se reconstruye con ``make_model`` + ``load_state_dict``.

    Attributes:
        parametrization: Siempre ``"epsilon"`` (se persiste en la receta del checkpoint).
    """

    def __init__(
        self,
        net: ScoreModel,
        marginal_std: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        """Inicializa el wrapper.

        Args:
            net: Red interna que predice en unidades de ε.
            marginal_std: Callable ``(x, t) -> std`` broadcastable ``(B, 1, …, 1)``, p. ej.
                ``lambda x, t: sde.marginal_prob(x, t)[1]`` (evita depender de ``diffusion.sde``).
        """
        super().__init__()
        self._net = net
        self._marginal_std = marginal_std
        self.parametrization = "epsilon"

    @property
    def inner(self) -> ScoreModel:
        """La red interna envuelta (la que predice en unidades de ε)."""
        return self._net

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Devuelve ``-inner(x, t) / clamp(marginal_std(x, t), 1e-5)``, con la shape de ``x``."""
        std = self._marginal_std(x, t)
        return -self._net(x, t) / std.clamp_min(_SIGMA_FLOOR)

    # Sin estos overrides, nn.Module serializaría las claves con prefijo "_net.".

    def state_dict(self, *args, **kwargs):
        """State dict del interno, con sus claves peladas (sin prefijo ``_net.``)."""
        return self._net.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Carga en el interno un state dict de red pelada (claves sin prefijo)."""
        return self._net.load_state_dict(state_dict, strict=strict, assign=assign)
