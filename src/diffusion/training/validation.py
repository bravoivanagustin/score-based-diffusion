"""Pérdida de validación por examen fijo.

:class:`FixedValExam` congela el examen (mismas imágenes, mismos ``t``, mismo ruido) re-sembrando
un ``torch.Generator`` local con :data:`VAL_EXAM_SEED` en cada evaluación, así la única variable
entre dos puntos de la curva es la red. No consume el RNG global ni el del loop, y no hay nada que
persistir al reanudar. El examen es específico del device: la serie solo es comparable dentro de
un mismo device. :func:`evaluate_with_weights` mide el mismo examen con otros pesos (p. ej. EMA).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final, TypedDict

import torch

from ..models import ScoreModel
from ..sde import ForwardSDE
from .losses import dsm_loss
from .time_sampling import TimeSampler

#: Semilla del examen fijo. Cambiarla cambia el examen y rompe la comparabilidad con curvas previas.
VAL_EXAM_SEED: Final[int] = 12345


class ValPoint(TypedDict):
    """Un punto de la serie dispersa de validación (una evaluación).

    Es un ``dict`` en runtime para que leer checkpoints y logs no dependa de esta clase.

    Attributes:
        step: Paso en que se midió, 1-indexado.
        raw: Pérdida de validación con los pesos vivos.
        ema: Ídem con la sombra EMA; ``None`` si la corrida no mantiene sombra.
        train: Examen fijo de entrenamiento con los pesos vivos; ``None`` sin esa fuente.
    """

    step: int
    raw: float
    ema: float | None
    train: float | None


class FixedValExam:
    """Examen fijo: la pérdida de DSM sobre un conjunto completo, congelada entre evaluaciones.

    Args:
        sde: Proceso forward de la corrida (el mismo con el que se entrena).
        batches: Fuente re-iterable de batches ``(B, *E)`` (p. ej. ``finite_batches``). Se
            recorre entera en cada evaluación; un iterador de un solo uso se agotaría.
        time_sampler: Muestreador de ``t`` ya construido; el mismo que usa el loop.
        device: Dispositivo donde se evalúa; ahí se mueven los batches y se crea el generator.
        seed: Semilla del examen; por defecto :data:`VAL_EXAM_SEED`.

    Attributes:
        device: El device normalizado a :class:`torch.device`.
        seed: La semilla con la que se re-siembra el generator en cada evaluación.
    """

    def __init__(
        self,
        sde: ForwardSDE,
        batches: Iterable[torch.Tensor],
        *,
        time_sampler: TimeSampler,
        device: torch.device | str = "cpu",
        seed: int = VAL_EXAM_SEED,
    ) -> None:
        self._sde = sde
        self._batches = batches
        self._time_sampler = time_sampler
        self.device = torch.device(device)
        self.seed = int(seed)

    def evaluate(self, net: ScoreModel) -> float:
        """Computa la pérdida del examen para la red recibida.

        Promedia sobre todos los batches ponderando por cantidad de imágenes
        (``Σ loss_batch · b / Σ b``), así el valor no depende del batcheo.

        Args:
            net: Red de score. Se evalúa en modo ``eval()`` sin gradientes y se restaura su modo
                al salir, incluso ante una excepción.

        Returns:
            La pérdida media por imagen, como ``float``.

        Raises:
            ValueError: Si la fuente no entregó ninguna imagen.
        """
        # Generator nuevo por evaluación, en el device del dato (torch exige que coincidan).
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)

        acumulado = 0.0  # Σ loss_batch · b
        imagenes = 0  # Σ b
        modo_previo = net.training
        net.eval()
        try:
            # Fuerza fp32 aun dentro de autocast: comparable entre corridas con y sin AMP.
            with (
                torch.no_grad(),
                torch.autocast(device_type=self.device.type, enabled=False),
            ):
                for batch in self._batches:
                    x0 = batch.to(self.device, non_blocking=True)
                    b = x0.shape[0]
                    if b == 0:
                        # Su media sería NaN; se saltea sin consumir el generator.
                        continue
                    t, sample_weights = self._time_sampler.sample(
                        b, generator=gen, device=self.device
                    )
                    loss = dsm_loss(
                        net, self._sde, x0, t, generator=gen, sample_weights=sample_weights
                    )
                    acumulado += loss.item() * b
                    imagenes += b
        finally:
            net.train(modo_previo)

        if imagenes == 0:
            raise ValueError(
                "el examen fijo no recibió ninguna imagen: la fuente de batches está vacía. "
                "Revisá que la carpeta del examen tenga imágenes y que la fuente sea "
                "re-iterable (un iterador de un solo uso queda agotado tras la primera "
                "evaluación)."
            )
        return acumulado / imagenes


def evaluate_with_weights(
    exam: FixedValExam, net: ScoreModel, weights: Mapping[str, torch.Tensor]
) -> float:
    """Corre el examen con otros pesos y deja la red exactamente como estaba.

    Clona los pesos vivos, carga ``weights``, evalúa y restaura (también ante una excepción).

    Args:
        exam: Examen fijo ya construido.
        net: Red de score a evaluar.
        weights: Pesos con exactamente las claves de ``net.state_dict()``. No se mutan.

    Returns:
        La pérdida del examen medida bajo ``weights``.

    Raises:
        RuntimeError: Si las claves de ``weights`` no coinciden con las de la red.
        ValueError: Si la fuente del examen no entregó imágenes.
    """
    # Clonar: state_dict() devuelve tensores vivos que load_state_dict sobreescribiría in-place.
    crudos = {k: v.detach().clone() for k, v in net.state_dict().items()}
    try:
        # Dentro del try: con claves que no coinciden, load_state_dict copia las que matchean
        # antes de levantar, y hay que restaurar igual.
        net.load_state_dict(weights)
        return exam.evaluate(net)
    finally:
        # Restauración in-place: las referencias del optimizador siguen siendo válidas.
        net.load_state_dict(crudos)
