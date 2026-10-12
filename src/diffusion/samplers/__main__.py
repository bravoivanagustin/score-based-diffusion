"""Smoke test manual del módulo: corre los 4 samplers sobre una red sin entrenar.

Correr con ``python -m diffusion.samplers``. Solo verifica que ``make_sampler -> sample``
corre y produce salidas finitas; no evalúa la calidad de las muestras.
"""

from __future__ import annotations

import torch

from diffusion.models import ScoreMLP
from diffusion.samplers import available_samplers, make_sampler
from diffusion.sde import make_sde


def main() -> dict[str, bool]:
    """Corre cada sampler del registry sobre una red sin entrenar y reporta finitud.

    Returns:
        ``{nombre_sampler: salida_finita}`` para todos los samplers del registry.
    """
    sde = make_sde("vp")
    net = ScoreMLP(data_dim=2)
    net.eval()

    def score_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return net(x, t)

    n_samples = 64
    summary: dict[str, bool] = {}
    for name in available_samplers():
        sampler = make_sampler(name, sde, score_fn, n_steps=20)
        x0 = sampler.sample(n_samples, generator=torch.Generator().manual_seed(0))
        is_finite = bool(torch.all(torch.isfinite(x0)))
        summary[name] = is_finite
        print(f"{name:7s} -> {tuple(x0.shape)}  |mean|~{x0.abs().mean():.3f}  finito={is_finite}")
    return summary


if __name__ == "__main__":
    main()
