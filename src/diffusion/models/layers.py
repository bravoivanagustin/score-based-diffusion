"""Piezas compartidas por las redes de score: registry de activaciones y embedding de tiempo."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

#: Funciones de activación soportadas, por nombre.
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def _make_activation(name: str) -> nn.Module:
    """Devuelve una nueva instancia de la activación ``name`` (p. ej. ``"silu"``).

    Raises:
        ValueError: Si ``name`` no está entre las activaciones soportadas.
    """
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError:
        opts = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(
            f"Activación desconocida '{name}'. Opciones: {opts}"
        ) from None


class SinusoidalEmbedding(nn.Module):
    """Embebe un escalar de tiempo ``t`` en un vector con senos y cosenos.

    Codificación posicional de Transformers, con senos y cosenos intercalados::

        embed(t)_{2i}   = sin(t · scale / 10000^{2i/d})
        embed(t)_{2i+1} = cos(t · scale / 10000^{2i/d})

    con ``i = 0, …, d/2 - 1`` y ``d = embed_dim``. Sin parámetros entrenables.

    Para ``t`` continuo en ``[0, 1]`` conviene ``scale=1000`` (como DDPM y
    ``score_sde_pytorch``, que usa ``t * 999``) para que se resuelvan tiempos chicos
    (``t ≲ 1e-2``).
    """

    def __init__(self, embed_dim: int = 128, scale: float = 1.0) -> None:
        """Inicializa el embedding.

        Args:
            embed_dim: Dimensión del vector de salida. Debe ser par (cada
                frecuencia aporta un seno y un coseno).
            scale: Factor aplicado a ``t`` antes de codificar; finito y positivo.

        Raises:
            ValueError: Si ``embed_dim`` no es par, o si ``scale`` no es
                finito y positivo.
        """
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim debe ser par (un seno y un coseno por frecuencia); "
                f"recibí embed_dim={embed_dim}"
            )
        scale = float(scale)
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(
                f"scale debe ser finito y positivo; recibí scale={scale}"
            )
        self.embed_dim = int(embed_dim)
        #: Factor de escala temporal (float plano, no buffer).
        self.scale = scale
        # Denominadores 10000^{2i/d} para i = 0 .. d/2 - 1, shape (d/2,).
        i = torch.arange(embed_dim // 2, dtype=torch.float32)
        denom = torch.pow(10000.0, (2.0 * i) / embed_dim)
        #: Buffer (no aprendible): viaja con .to(device).
        self.register_buffer("denom", denom)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embebe el tiempo.

        Args:
            t: Tensor de tiempos de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Tensor de shape ``(B, embed_dim)`` con senos y cosenos intercalados.
        """
        t = t.reshape(-1) * self.scale  # (B, 1) o (B,) -> (B,), escalado
        # args[b, i] = t_b * scale / denom_i  ->  (B, d/2)
        args = t[:, None] / self.denom[None, :]
        # Intercalar sin/cos: stack -> (B, d/2, 2) -> reshape (B, d).
        emb = torch.stack((torch.sin(args), torch.cos(args)), dim=-1)
        return emb.reshape(t.shape[0], self.embed_dim)
