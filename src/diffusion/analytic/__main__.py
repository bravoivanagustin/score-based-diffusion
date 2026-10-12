"""Smoke test del oráculo en CPU: score cerrado vs. autograd y masa unitaria.

Para VP, VE y sub-VP sobre una mixtura chica (pesos desbalanceados, una componente
rotada) compara el score en forma cerrada con ``autograd`` de ``log_prob`` y verifica
que la densidad integre a uno. Correr con ``python -m diffusion.analytic``.
"""

from __future__ import annotations

import math

import torch

from diffusion.analytic import MixtureOracle
from diffusion.data_generation import ExactGaussianMixture
from diffusion.sde import make_sde

_SDES = ("vp", "ve", "sub_vp")
_TIEMPOS = (0.05, 0.5, 1.0)
_N_PUNTOS = 256


def _mixtura() -> ExactGaussianMixture:
    """Mixtura exacta chica: pesos desbalanceados y una componente rotada y estirada."""
    angulo = math.pi / 3.0
    rotacion = torch.tensor(
        [[math.cos(angulo), -math.sin(angulo)], [math.sin(angulo), math.cos(angulo)]],
        dtype=torch.float64,
    )
    escalas = torch.diag(torch.tensor([1.0, 0.09], dtype=torch.float64))
    rotada = rotacion @ escalas @ rotacion.T
    return ExactGaussianMixture(
        2,
        weights=[0.3, 0.7],
        means=[[-1.5, 0.5], [2.0, -1.0]],
        covariances=[rotada.tolist(), [[0.25, 0.0], [0.0, 0.04]]],
        seed=0,
    )


def _discrepancia_contra_autograd(
    oraculo: MixtureOracle, x: torch.Tensor, instante: float
) -> float:
    """Máxima diferencia absoluta entre el score cerrado y el gradiente por autograd."""
    tiempos = torch.full((x.shape[0],), instante, dtype=x.dtype)
    puntos = x.clone().requires_grad_(True)
    (gradiente,) = torch.autograd.grad(oraculo.log_prob(puntos, tiempos).sum(), puntos)
    cerrado = oraculo.score(x, tiempos)
    return float((cerrado - gradiente).abs().max())


def main() -> dict[str, tuple[float, float]]:
    """Corre ambos chequeos sobre las tres SDEs e imprime los resultados.

    Returns:
        ``{nombre_sde: (discrepancia_maxima, masa_mas_alejada_de_uno)}``, el peor
        caso de cada SDE sobre todos los tiempos.
    """
    mixtura = _mixtura()
    x = torch.randn(_N_PUNTOS, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    resumen: dict[str, tuple[float, float]] = {}
    for nombre in _SDES:
        oraculo = MixtureOracle(mixtura, make_sde(nombre))
        peor_discrepancia = 0.0
        peor_masa = 1.0
        for instante in _TIEMPOS:
            discrepancia = _discrepancia_contra_autograd(oraculo, x, instante)
            masa = oraculo.total_mass(instante)
            peor_discrepancia = max(peor_discrepancia, discrepancia)
            if abs(masa - 1.0) > abs(peor_masa - 1.0):
                peor_masa = masa
            # Solo ASCII: la consola de Windows (cp1252) no encodea símbolos matemáticos.
            print(
                f"{nombre:7s} t={instante:<5.3g} score vs autograd: "
                f"max|dif|={discrepancia:.3e}   masa=int p_t={masa:.12f}"
            )
        resumen[nombre] = (peor_discrepancia, peor_masa)
    return resumen


if __name__ == "__main__":
    main()
