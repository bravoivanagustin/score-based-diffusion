"""Red de score U-Net condicionada en el tiempo, para imágenes ``(B, C, H, W)``.

Define :class:`ScoreUNet` y sus bloques internos. Determinística (GroupNorm, sin dropout
ni batchnorm).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SinusoidalEmbedding, _make_activation


class TimeMLP(nn.Module):
    """Proyección del tiempo al vector de condicionamiento de la U-Net.

    ``SinusoidalEmbedding -> Linear -> activación -> Linear``. Se computa una vez por
    forward y la comparten todos los bloques.
    """

    def __init__(
        self,
        embed_dim: int,
        time_embed_dim: int,
        activation: str = "silu",
        time_scale: float = 1.0,
    ) -> None:
        """Inicializa la proyección temporal.

        Args:
            embed_dim: Dimensión del embedding sinusoidal (par).
            time_embed_dim: Dimensión del vector de condicionamiento de salida.
            activation: Nombre de la activación entre las dos lineales.
            time_scale: Factor de escala temporal del
                :class:`~diffusion.models.layers.SinusoidalEmbedding`.

        Raises:
            ValueError: Si ``time_scale`` no es finito y positivo.
        """
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.time_embed_dim = int(time_embed_dim)

        self.embed = SinusoidalEmbedding(embed_dim, scale=time_scale)
        #: Escala temporal ya validada por el embedding.
        self.time_scale = self.embed.scale
        self.lin1 = nn.Linear(embed_dim, time_embed_dim)
        self.act = _make_activation(activation)
        self.lin2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Proyecta ``t`` ``(B,)`` o ``(B, 1)`` a ``(B, time_embed_dim)``."""
        return self.lin2(self.act(self.lin1(self.embed(t))))


class ConvResBlock(nn.Module):
    """Bloque residual convolucional con inyección aditiva del tiempo.

    El flujo es::

        GroupNorm -> activación -> Conv 3x3
        + proyección lineal de t_emb  (broadcast sobre H, W)
        GroupNorm -> activación -> Conv 3x3
        + skip  (identidad, o conv 1x1 si cambian los canales)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        groups: int = 8,
        activation: str = "silu",
    ) -> None:
        """Inicializa el bloque.

        Args:
            in_channels: Canales del tensor de entrada.
            out_channels: Canales del tensor de salida.
            time_embed_dim: Dimensión de ``t_emb`` (salida de :class:`TimeMLP`).
            groups: Grupos de GroupNorm; debe dividir a ``in_channels`` y ``out_channels``.
            activation: Nombre de la activación.
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.time_embed_dim = int(time_embed_dim)

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = _make_activation(activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Se suma tras conv1.
        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = _make_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Aplica el bloque: ``(B, in_channels, H, W) -> (B, out_channels, H, W)``.

        Args:
            x: Tensor de shape ``(B, in_channels, H, W)``.
            t_emb: Vector temporal de shape ``(B, time_embed_dim)``.
        """
        h = self.conv1(self.act1(self.norm1(x)))       # (B, out_channels, H, W)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))        # (B, out_channels, H, W)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Auto-atención espacial *single-head* con conexión residual.

    Trata las ``H·W`` posiciones como tokens de dimensión ``C``; se usa en resoluciones
    bajas, donde ``H·W`` es manejable. Preserva la shape ``(B, C, H, W)``::

        GroupNorm -> QKV (conv 1x1) -> scaled_dot_product_attention -> conv 1x1 -> + skip
    """

    def __init__(self, channels: int, groups: int = 8) -> None:
        """Inicializa el bloque de atención.

        Args:
            channels: Canales de entrada y salida.
            groups: Grupos de GroupNorm; debe dividir a ``channels``.
        """
        super().__init__()
        self.channels = int(channels)

        self.norm = nn.GroupNorm(groups, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica la auto-atención: ``(B, channels, H, W) -> (B, channels, H, W)``."""
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.to_qkv(h).chunk(3, dim=1)
        # (B, C, H, W) -> (B, H*W, C).
        q = q.reshape(B, C, H * W).transpose(1, 2)
        k = k.reshape(B, C, H * W).transpose(1, 2)
        v = v.reshape(B, C, H * W).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)   # (B, H*W, C)
        attn = attn.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj_out(attn)


class Downsample(nn.Module):
    """Reducción espacial ×2 por convolución ``3x3`` con stride 2 (conserva canales)."""

    def __init__(self, channels: int) -> None:
        """Inicializa la reducción con ``channels`` canales de entrada y salida."""
        super().__init__()
        self.channels = int(channels)
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce ``(B, C, H, W)`` (``H``, ``W`` pares) a ``(B, C, H // 2, W // 2)``."""
        return self.conv(x)


class Upsample(nn.Module):
    """Ampliación espacial ×2 por interpolación *nearest* + convolución ``3x3``.

    Evita el *checkerboard* de la convolución transpuesta. Conserva canales.
    """

    def __init__(self, channels: int) -> None:
        """Inicializa la ampliación con ``channels`` canales de entrada y salida."""
        super().__init__()
        self.channels = int(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Amplía ``(B, C, H, W)`` a ``(B, C, H * 2, W * 2)``."""
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ScoreUNet(nn.Module):
    """Red de score U-Net estilo DDPM (Ho et al. 2020; Song et al. 2021) para imágenes.

    - **Entrada**: convolución ``3x3`` de ``in_channels`` a ``base_channels``.
    - **Encoder**: un nivel por entrada de ``channel_mults``; el nivel ``i`` trabaja
      con ``base_channels * channel_mults[i]`` canales y ``num_res_blocks``
      :class:`ConvResBlock`, con :class:`AttentionBlock` en los niveles cuya
      resolución (``image_size / 2**i``) pertenece a ``attn_resolutions``. Entre
      niveles hay un :class:`Downsample` (``len(channel_mults) - 1`` en total). Cada
      activación del encoder se guarda como *skip*.
    - **Bottleneck**: ``ConvResBlock → AttentionBlock → ConvResBlock``.
    - **Decoder**: espejo del encoder; antes de cada :class:`ConvResBlock` se
      **concatena por canales** el *skip* correspondiente del encoder, y se
      :class:`Upsample` al final de cada nivel para volver a la resolución de
      entrada.
    - **Salida**: ``GroupNorm → activación → Conv 3x3`` a ``in_channels``, sin
      activación final (el score es no acotado).
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 64,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        embed_dim: int = 128,
        time_embed_dim: int = 256,
        attn_resolutions: tuple[int, ...] = (16,),
        groups: int = 8,
        activation: str = "silu",
        time_scale: float = 1.0,
    ) -> None:
        """Inicializa la red (los defaults son la arquitectura de referencia).

        Args:
            in_channels: Canales de la imagen de entrada y de la salida (p. ej. 1
                para escala de grises, 3 para RGB).
            image_size: Resolución espacial (``H == W``); fija la resolución de cada
                nivel (``image_size / 2**i``) y con ella dónde va la atención.
            base_channels: Canales del primer nivel (los demás son
                ``base_channels * channel_mults[i]``).
            channel_mults: Multiplicador de canales por nivel; su longitud es la
                cantidad de niveles del encoder/decoder.
            num_res_blocks: Cantidad de :class:`ConvResBlock` por nivel del encoder.
            embed_dim: Dimensión del embedding sinusoidal de tiempo (par).
            time_embed_dim: Dimensión del vector de condicionamiento de :class:`TimeMLP`.
            attn_resolutions: Resoluciones espaciales (absolutas) en las que se
                coloca :class:`AttentionBlock`; el bottleneck siempre la lleva.
            groups: Grupos de GroupNorm; debe dividir a todos los anchos de canal.
            activation: Nombre de la activación.
            time_scale: Factor de escala temporal del
                :class:`~diffusion.models.layers.SinusoidalEmbedding`.

        Raises:
            ValueError: Si ``image_size`` no es divisible por ``2**(len(channel_mults)-1)``,
                si ``groups`` no divide a algún ancho de canal, o si ``time_scale`` no es
                finito y positivo.
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.base_channels = int(base_channels)
        self.channel_mults = tuple(int(m) for m in channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.embed_dim = int(embed_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.attn_resolutions = tuple(int(r) for r in attn_resolutions)
        self.groups = int(groups)

        # Validar antes de construir, para no fallar con un error críptico de torch.
        reduction = 2 ** (len(self.channel_mults) - 1)
        if self.image_size % reduction != 0:
            raise ValueError(
                f"image_size debe ser divisible por {reduction} "
                f"(= 2**(len(channel_mults)-1), un Downsample por nivel salvo el "
                f"último); recibí image_size={self.image_size}"
            )
        for level, mult in enumerate(self.channel_mults):
            width = self.base_channels * mult
            if width % self.groups != 0:
                raise ValueError(
                    f"groups debe dividir a todos los anchos de canal; en el nivel "
                    f"{level} el ancho es base_channels*{mult}={width}, no divisible "
                    f"por groups={self.groups}"
                )

        num_levels = len(self.channel_mults)
        attn_set = set(self.attn_resolutions)

        self.time_mlp = TimeMLP(embed_dim, time_embed_dim, activation, time_scale)
        #: Escala temporal ya validada por el embedding.
        self.time_scale = self.time_mlp.time_scale

        self.conv_in = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )

        # Encoder. skip_channels registra los canales de cada skip para dimensionar el decoder.
        self.down_blocks = nn.ModuleList()
        ch = self.base_channels
        skip_channels: list[int] = [ch]
        for level, mult in enumerate(self.channel_mults):
            out_ch = self.base_channels * mult
            resolution = self.image_size // (2 ** level)
            use_attn = resolution in attn_set
            for _ in range(self.num_res_blocks):
                stage: list[nn.Module] = [
                    ConvResBlock(ch, out_ch, time_embed_dim, groups, activation)
                ]
                ch = out_ch
                if use_attn:
                    stage.append(AttentionBlock(ch, groups))
                self.down_blocks.append(nn.ModuleList(stage))
                skip_channels.append(ch)
            if level != num_levels - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)

        # Bottleneck.
        self.mid_block1 = ConvResBlock(ch, ch, time_embed_dim, groups, activation)
        self.mid_attn = AttentionBlock(ch, groups)
        self.mid_block2 = ConvResBlock(ch, ch, time_embed_dim, groups, activation)

        # Decoder: cada nivel consume num_res_blocks + 1 skips (concatenados por canales).
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(num_levels)):
            out_ch = self.base_channels * self.channel_mults[level]
            resolution = self.image_size // (2 ** level)
            use_attn = resolution in attn_set
            for i in range(self.num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                stage = [
                    ConvResBlock(
                        ch + skip_ch, out_ch, time_embed_dim, groups, activation
                    )
                ]
                ch = out_ch
                if use_attn:
                    stage.append(AttentionBlock(ch, groups))
                if level != 0 and i == self.num_res_blocks:
                    stage.append(Upsample(ch))
                self.up_blocks.append(nn.ModuleList(stage))

        # Cabeza de salida, sin activación final.
        self.out_norm = nn.GroupNorm(groups, ch)
        self.out_act = _make_activation(activation)
        self.conv_out = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    @staticmethod
    def _apply_stage(
        stage: nn.ModuleList, x: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        """Aplica en orden las capas de una etapa; solo los ConvResBlock reciben ``t_emb``."""
        for layer in stage:
            if isinstance(layer, ConvResBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predice el score en ``(x, t)``.

        Args:
            x: Imagen ruidosa de shape ``(B, in_channels, image_size, image_size)``.
            t: Tiempo / nivel de ruido de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Score predicho con la misma shape que ``x``.

        Raises:
            ValueError: Si el alto/ancho de ``x`` no es ``image_size`` o sus canales no
                son ``in_channels``.
        """
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            raise ValueError(
                f"x debe tener alto y ancho iguales a image_size={self.image_size} "
                f"(la arquitectura quedó fijada en construcción); recibí "
                f"(H, W)=({x.shape[-2]}, {x.shape[-1]})"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"x debe tener in_channels={self.in_channels} canales; "
                f"recibí {x.shape[1]}"
            )

        t_emb = self.time_mlp(t)               # (B, time_embed_dim)

        # Encoder.
        h = self.conv_in(x)
        skips: list[torch.Tensor] = [h]
        for stage in self.down_blocks:
            h = self._apply_stage(stage, h, t_emb)
            skips.append(h)

        # Bottleneck.
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder.
        for stage in self.up_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = self._apply_stage(stage, h, t_emb)

        h = self.out_act(self.out_norm(h))
        return self.conv_out(h)


if __name__ == "__main__":
    # Smoke test manual: python -m diffusion.models.unet
    net = ScoreUNet()
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    out = net(x, t)
    print(f"ScoreUNet(in_channels=3, image_size=64): salida {tuple(out.shape)}")

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")
