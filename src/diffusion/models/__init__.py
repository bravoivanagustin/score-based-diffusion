"""Redes de score :math:`s_\\theta(x, t) \\approx \\nabla_x \\log p_t(x)` y registry por nombre.

Incluye :class:`ScoreMLP` (datos 2D), :class:`ScoreUNet` (imágenes), el Protocol
:class:`ScoreModel` y :class:`EpsilonScoreWrapper` (parametrización ε del score).

Uso típico::

    from diffusion.models import ScoreMLP, make_model

    net = ScoreMLP(data_dim=2)
    score = net(x, t)

    net2 = make_model("mlp", data_dim=2)   # misma red vía registry
"""

from __future__ import annotations

import inspect

from .base import ScoreModel
from .layers import SinusoidalEmbedding
from .mlp import ResidualBlock, ScoreMLP
from .parametrization import EpsilonScoreWrapper
from .unet import ScoreUNet

REGISTRY: dict[str, type] = {
    "mlp": ScoreMLP,
    "unet": ScoreUNet,
}


def available_models() -> list[str]:
    """Nombres de las redes disponibles, ordenados."""
    return sorted(REGISTRY)


def make_model(name: str, **kwargs) -> ScoreModel:
    """Crea la red de score ``name``.

    Los ``kwargs`` que no figuran en la firma del constructor se descartan, así un caller
    genérico (config o reconstrucción desde checkpoint) puede pasar siempre el mismo conjunto.
    """
    try:
        cls = REGISTRY[name]
    except KeyError:
        opts = ", ".join(available_models())
        raise ValueError(f"Red desconocida '{name}'. Opciones: {opts}") from None
    params = inspect.signature(cls).parameters
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    clean = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in params}
    return cls(**clean)


__all__ = [
    "ScoreModel",
    "SinusoidalEmbedding",
    "ResidualBlock",
    "ScoreMLP",
    "ScoreUNet",
    "EpsilonScoreWrapper",
    "REGISTRY",
    "available_models",
    "make_model",
]
