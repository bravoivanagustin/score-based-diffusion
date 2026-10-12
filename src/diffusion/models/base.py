"""Protocol :class:`ScoreModel`: el contrato ``(x, t) -> score`` de las redes de score.

Es tipado estructural: las redes lo satisfacen por su firma, sin heredar de él.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ScoreModel(Protocol):
    """Una red de score: callable ``(x, t) -> score`` con ``score.shape == x.shape``.

    ``x`` es el dato ruidoso (``(B, data_dim)`` o ``(B, C, H, W)``) y ``t`` el tiempo de
    shape ``(B,)`` o ``(B, 1)``.
    """

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...
