"""Smoke test manual: ``python -m diffusion.sde``.

Para cada SDE del registry imprime media/desvío del kernel en ``t ≈ 0`` (≈ ``x0``, desvío
≈ 0) y en ``t = T`` (≈ prior).
"""

from __future__ import annotations

import torch

from diffusion.sde import available_sdes, make_sde


def main() -> None:
    x0 = torch.randn(16, 2)  # batch dummy (B, 2)
    for name in available_sdes():
        sde = make_sde(name)
        t = torch.rand(16)
        x_t, _ = sde.perturb(x0, t)
        print(f"{name:7s} data_dim={sde.data_dim}  perturb -> {tuple(x_t.shape)}")
        for tv in (1e-3, sde.T):
            tt = torch.full((16,), float(tv))
            mean, scale = sde.marginal_prob(x0, tt)
            print(f"    t={tv:<6.3g} |mean|~{mean.abs().mean():.3f}  std~{scale.mean():.3f}")


if __name__ == "__main__":
    main()
