"""Densidad, score y sesgo exactos de una mixtura gaussiana 2D bajo SDEs escalares.

Para una mixtura de parámetros conocidos calcula ``p_t(x)``, ``log p_t(x)`` y
``∇_x log p_t(x)`` en forma cerrada para VP, VE y sub-VP, más el sesgo de
inicialización. El oráculo cumple el contrato ``(x, t) -> score`` de los samplers.

Uso típico::

    from diffusion.analytic import MixtureOracle, auto_grid, integrate

    oraculo = MixtureOracle(mixtura, make_sde("vp"))
    score = oraculo(x, t)
    masa = oraculo.total_mass(0.5)        # autochequeo por cuadratura: debe dar 1

Los nombres se resuelven de forma diferida (PEP 562) para que importar
``diffusion.analytic.quadrature`` no arrastre :mod:`diffusion.sde` ni
:mod:`diffusion.data_generation`.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mixture_oracle import BiasReport, MixtureOracle
    from .quadrature import QuadratureGrid, auto_grid, integrate

__all__ = [
    "BiasReport",
    "MixtureOracle",
    "QuadratureGrid",
    "auto_grid",
    "integrate",
]

# Nombre público -> submódulo que lo define.
_EXPORTS: dict[str, str] = {
    "BiasReport": "mixture_oracle",
    "MixtureOracle": "mixture_oracle",
    "QuadratureGrid": "quadrature",
    "auto_grid": "quadrature",
    "integrate": "quadrature",
}


def __getattr__(name: str) -> Any:
    """Importa el submódulo de ``name`` en el primer acceso y cachea el objeto."""
    submodulo = _EXPORTS.get(name)
    if submodulo is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    valor = getattr(import_module(f".{submodulo}", __name__), name)
    globals()[name] = valor
    return valor


def __dir__() -> list[str]:
    """Incluye la API pública aún no resuelta, para autocompletado."""
    return sorted(set(globals()) | set(__all__))
