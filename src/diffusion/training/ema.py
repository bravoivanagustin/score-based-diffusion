"""Sombra EMA de los pesos: la media móvil exponencial que publican los checkpoints.

.. math::

    \\text{ema}_s = d_s \\cdot \\text{ema}_{s-1} + (1 - d_s)\\cdot \\theta_s,
    \\qquad d_s = \\min\\!\\left(d, \\frac{1+s}{10+s}\\right)

``s`` es la cantidad de pasos completados (1-indexado): se llama ``update(s)`` justo después del
``optimizer.step()`` número ``s`` y la sombra arranca en ``ema_0 = θ_0``. El factor
``(1+s)/(10+s)`` es una rampa de warmup (``s=1`` da ``≈0.18``; alcanza ``d=0.999`` en
``s ≈ 8990``). Cambiar esta convención rompe la comparabilidad entre corridas.

La sombra es pasiva (solo lee los pesos, no consume RNG) y se aplica a los tensores entrenables;
los buffers se copian del módulo vivo al publicar (exacto mientras sean constantes).
"""

from __future__ import annotations

import math

import torch


class EmaShadow:
    """Media móvil exponencial de los pesos de un módulo, con rampa de warmup.

    Las claves son las del ``state_dict`` del módulo (no las de ``named_parameters``), así
    :meth:`state_dict` es cargable con ``strict=True`` en una red reconstruida por la receta,
    incluso si el módulo es un ``EpsilonScoreWrapper``.

    Args:
        module: Módulo a observar (cualquier :class:`~diffusion.models.ScoreModel`); nunca se
            modifica.
        decay: Decay del EMA ``d``; finito y en el intervalo abierto ``(0, 1)`` (p. ej. 0.999).

    Attributes:
        decay: El decay configurado (techo de la rampa de warmup).

    Raises:
        ValueError: Si ``decay`` no es finito o cae fuera de ``(0, 1)``, o si no se pudo
            rastrear ningún tensor entrenable del módulo.
    """

    def __init__(self, module: torch.nn.Module, decay: float) -> None:
        self.decay = _validar_decay(decay)
        self._module = module
        entrenables = {id(p) for p in module.parameters() if p.requires_grad}
        vivos = module.state_dict(keep_vars=True)
        # Filtra por identidad (no por nombre) para no depender de cómo el módulo arme sus claves.
        self._nombres = [nombre for nombre, t in vivos.items() if id(t) in entrenables]
        if entrenables and not self._nombres:
            # Típicamente un override de state_dict que no reenvía keep_vars: sin este guard la
            # sombra quedaría vacía y el checkpoint publicaría pesos crudos como si fueran EMA.
            raise ValueError(
                f"EmaShadow no pudo rastrear ningún tensor entrenable de {type(module).__name__}: "
                f"tiene {len(entrenables)} parámetro(s) entrenable(s) pero su state_dict no "
                "devolvió los tensores vivos. Revisá que el módulo reenvíe keep_vars=True a "
                "nn.Module.state_dict (la sombra los identifica por identidad)."
            )
        self._sombra = {nombre: vivos[nombre].detach().clone() for nombre in self._nombres}

    def update(self, step: int) -> None:
        """Actualiza la sombra; se llama después del ``optimizer.step()`` número ``step``.

        Args:
            step: Pasos completados, 1-indexado. Fija ``d_s = min(decay, (1+step)/(10+step))``.

        Raises:
            ValueError: Si ``step < 1``.
        """
        if step < 1:
            raise ValueError(
                "step es la cantidad de pasos completados (1-indexado) y debe ser >= 1; "
                f"se recibió step={step}"
            )
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        vivos = self._module.state_dict(keep_vars=True)
        with torch.no_grad():
            for nombre in self._nombres:
                self._sombra[nombre].mul_(d).add_(vivos[nombre].detach(), alpha=1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Foto publicable de la sombra: el ``state_dict`` completo del módulo con EMA.

        Returns:
            Un dict con las mismas claves que ``module.state_dict()``: los tensores entrenables
            con su valor EMA y el resto copiado del módulo vivo. Todos son clones, así que la
            foto no cambia si la sombra sigue actualizándose.
        """
        vivos = self._module.state_dict(keep_vars=True)
        return {nombre: self._sombra.get(nombre, t).detach().clone() for nombre, t in vivos.items()}

    def load_state(self, shadow: dict[str, torch.Tensor]) -> None:
        """Restaura la sombra desde una foto previa (reanudación de una corrida).

        Args:
            shadow: Foto producida por :meth:`state_dict`; las claves extra (buffers) se
                ignoran.

        Raises:
            ValueError: Si falta alguna de las claves rastreadas por la sombra.
        """
        faltantes = [nombre for nombre in self._nombres if nombre not in shadow]
        if faltantes:
            raise ValueError(
                "la foto de la sombra EMA no tiene todas las claves entrenables del módulo; "
                f"faltan: {', '.join(faltantes)}"
            )
        with torch.no_grad():
            for nombre in self._nombres:
                self._sombra[nombre].copy_(shadow[nombre])


def _validar_decay(decay: float) -> float:
    """Valida el decay del EMA (finito, en ``(0, 1)``) y lo devuelve como ``float``.

    Raises:
        ValueError: Si no es convertible a float, no es finito o cae fuera de ``(0, 1)``.
    """
    mensaje = (
        "ema_decay debe ser finito y estar en el intervalo abierto (0, 1); "
        f"se recibió decay={decay}"
    )
    try:
        valor = float(decay)
    except (TypeError, ValueError):
        raise ValueError(mensaje) from None
    if not math.isfinite(valor) or not 0.0 < valor < 1.0:
        raise ValueError(mensaje)
    return valor
