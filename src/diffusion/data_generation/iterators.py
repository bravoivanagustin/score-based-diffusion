"""Adaptadores de iteración sobre las fuentes de datos de juguete.

El loop de entrenamiento consume una fuente infinita de tensores ``(B, ...)``;
:func:`infinite_bare` adapta el ``DataLoader`` finito de 1-tuplas ``(x0,)``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def infinite_bare(loader) -> Iterator[torch.Tensor]:
    """Recorre un ``DataLoader`` finito en bucle y yield-ea el tensor crudo.

    Args:
        loader: ``DataLoader`` (u otro iterable) que yield-ea 1-tuplas ``(x0,)``.

    Yields:
        El tensor crudo ``x0`` de cada batch, indefinidamente.
    """
    while True:
        for (x0,) in loader:
            yield x0
