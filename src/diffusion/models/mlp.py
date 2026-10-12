"""Red de score MLP condicionada en el tiempo, para datos 2D.

Determinística (sin dropout ni batchnorm): la estocasticidad vive en la SDE y el sampler.

Uso típico::

    from diffusion.models import ScoreMLP

    net = ScoreMLP(data_dim=2)          # VP / VE / sub-VP
    score = net(x, t)                    # x: (B, 2), t: (B,) -> (B, 2)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import SinusoidalEmbedding, _make_activation


class ResidualBlock(nn.Module):
    """Bloque MLP residual: dos lineales con activación y skip identidad.

    ``salida = Linear(act(Linear(x))) + x``.
    """

    def __init__(self, hidden_dim: int, activation: str = "silu") -> None:
        """Inicializa el bloque.

        Args:
            hidden_dim: Ancho de las capas lineales (entrada = salida).
            activation: Nombre de la activación entre las dos lineales.
        """
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = _make_activation(activation)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica el bloque: ``(B, hidden_dim) -> (B, hidden_dim)``."""
        return self.lin2(self.act(self.lin1(x))) + x


class ScoreMLP(nn.Module):
    """Red de score completa: embedding de tiempo + bloques residuales.

    Concatena ``x`` con el embedding de ``t``, proyecta a ``hidden_dim``, pasa por
    ``num_blocks`` :class:`ResidualBlock` y proyecta a ``data_dim`` sin activación final
    (el score es no acotado).
    """

    def __init__(
        self,
        data_dim: int = 2,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        activation: str = "silu",
        time_scale: float = 1.0,
    ) -> None:
        """Inicializa la red.

        Args:
            data_dim: Dimensión del dato de entrada y de la salida (el score).
            embed_dim: Dimensión del embedding de tiempo.
            hidden_dim: Ancho de las capas ocultas en todos los bloques.
            num_blocks: Cantidad de bloques residuales.
            activation: Nombre de la activación (se pasa a cada bloque y a la
                proyección de entrada).
            time_scale: Factor de escala temporal del
                :class:`~diffusion.models.layers.SinusoidalEmbedding`.

        Raises:
            ValueError: Si ``time_scale`` no es finito y positivo.
        """
        super().__init__()
        self.data_dim = int(data_dim)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)

        self.time_embed = SinusoidalEmbedding(embed_dim, scale=time_scale)
        #: Escala temporal ya validada por el embedding.
        self.time_scale = self.time_embed.scale
        self.input_proj = nn.Linear(data_dim + embed_dim, hidden_dim)
        self.input_act = _make_activation(activation)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, activation) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predice el score en ``(x, t)``.

        Args:
            x: Dato ruidoso de shape ``(B, data_dim)``.
            t: Tiempo / nivel de ruido de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Score predicho de shape ``(B, data_dim)``.
        """
        t_emb = self.time_embed(t)                  # (B, embed_dim)
        h = torch.cat([x, t_emb], dim=-1)           # (B, data_dim + embed_dim)
        h = self.input_act(self.input_proj(h))      # (B, hidden_dim)
        h = self.blocks(h)                          # (B, hidden_dim)
        return self.output_proj(h)                  # (B, data_dim)


if __name__ == "__main__":
    # Smoke test manual: python -m diffusion.models.mlp
    net = ScoreMLP()
    x = torch.randn(16, 2)
    t = torch.rand(16)
    out = net(x, t)
    print(f"ScoreMLP(data_dim=2): salida {tuple(out.shape)}")

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")
