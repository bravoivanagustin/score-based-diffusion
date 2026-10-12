"""Samplers del proceso reverso: integran la SDE/ODE inversa a partir del score.

Expone el ABC :class:`~diffusion.samplers.base.ReverseSampler`, los samplers concretos
(Euler–Maruyama, PF-ODE, Heun, predictor–corrector), el registry/factory por nombre y
la generación desde checkpoint.

Uso típico::

    sde = make_sde("vp")
    sampler = make_sampler("euler", sde, score_fn)
    x0 = sampler.sample(64)
"""

from __future__ import annotations

import inspect

from diffusion.sde import ForwardSDE

from .base import ReverseSampler, ScoreFn
from .euler_maruyama import EulerMaruyama
from .heun import HeunODE
from .pf_ode import ProbabilityFlowODE
from .predictor_corrector import PredictorCorrector
from .time_grid import (
    TimeGrid,
    TimeGridFn,
    available_time_grids,
    make_time_grid,
)

REGISTRY: dict[str, type[ReverseSampler]] = {
    cls.name: cls for cls in (EulerMaruyama, ProbabilityFlowODE, HeunODE, PredictorCorrector)
}


def available_samplers() -> list[str]:
    """Nombres de los samplers disponibles, ordenados."""
    return sorted(REGISTRY)


def make_sampler(name: str, sde: ForwardSDE, score_fn: ScoreFn, **kwargs) -> ReverseSampler:
    """Crea el sampler ``name`` configurado con ``sde`` y ``score_fn``.

    Los ``kwargs`` que no aplican al sampler elegido se descartan (filtrado por la firma
    del constructor), así un caller genérico puede pasar siempre el mismo conjunto.

    Args:
        name: Clave del registry (``"euler"``, ``"pf_ode"``, ``"heun"``, ``"pc"``).
        sde: Proceso forward del que el sampler deriva ``(f, g)`` y el prior.
        score_fn: Función pura ``(x, t) -> score`` que aproxima ``∇_x log p_t(x)``.
        **kwargs: Parámetros opcionales (``n_steps``, ``t_eps``, ``time_grid``; para
            ``pc`` también ``n_corrector``/``snr``).

    Raises:
        ValueError: Si ``name`` no está en el registry.
    """
    try:
        cls = REGISTRY[name]
    except KeyError:
        opts = ", ".join(available_samplers())
        raise ValueError(f"Sampler desconocido '{name}'. Opciones: {opts}") from None
    params = inspect.signature(cls).parameters
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    clean = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in params}
    return cls(sde, score_fn, **clean)


# Importado al final: ``generate`` depende de ``make_sampler``.
from .generate import generate_from_checkpoint, load_score_model  # noqa: E402

__all__ = [
    "ReverseSampler",
    "ScoreFn",
    "REGISTRY",
    "available_samplers",
    "make_sampler",
    "generate_from_checkpoint",
    "load_score_model",
    "EulerMaruyama",
    "ProbabilityFlowODE",
    "HeunODE",
    "PredictorCorrector",
    "TimeGrid",
    "TimeGridFn",
    "available_time_grids",
    "make_time_grid",
]
