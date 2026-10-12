"""Capa de configuración: de un archivo YAML a una corrida lista para :func:`train`.

:func:`load_config` lee el YAML a un ``dict`` y :func:`build_run` lo ensambla en un
:class:`RunSpec` usando los factories ``make_sde``/``make_distribution``/``make_model``.

Estructura esperada del YAML::

    sde:                  # -> make_sde(name, **resto)
      name: vp
      beta_min: 0.1
    data:                 # -> make_distribution(shape, dim, **resto)
      shape: mixture
      dim: 2
      n_samples: 4000     # n_samples/batch_size describen la fuente (no van al TrainConfig)
      batch_size: 256
      n_components: 8
      val_root: data/cats_val  # opcional, solo kind: images: activa la pérdida de validación
                          #   y el examen fijo de entrenamiento (desde `root`).
    train:                # -> campos de TrainConfig
      num_steps: 300
      lr: 0.002
      checkpoint_every: 50  # opcional: 0 = solo final; N>0 = snapshots …_stepNNNNN.pt
    model:                # opcional: -> make_model(name, **resto); default mlp/unet
      name: mlp
      hidden_dim: 256
      score_parametrization: epsilon  # opcional: envuelve la red con EpsilonScoreWrapper
    out:                  # rutas de salida (relativas al cwd)
      checkpoint: models/vp_mixture.pt
      loss_curve: models/vp_mixture_loss.png
      train_log: models/vp_mixture_log.jsonl  # opcional
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields

from ..data_generation import (
    count_images,
    finite_batches,
    infinite_bare,
    infinite_batches,
    make_distribution,
)
from ..models import EpsilonScoreWrapper, ScoreModel, make_model
from ..sde import ForwardSDE, make_sde
from .trainer import TrainConfig

# Defaults de la fuente de datos cuando el bloque ``data:`` no los especifica.
_DEFAULT_N_SAMPLES = 4000
_DEFAULT_BATCH_SIZE = 256

# Valores válidos de ``data.kind``; ``points`` es el default.
_VALID_KINDS = ("points", "images")


@dataclass
class RunSpec:
    """Una corrida lista: SDE + red + fuente de datos + hiperparámetros + rutas de salida.

    ``model_spec`` es la receta ``{name, kwargs}`` de la red, para que el checkpoint sea
    reconstruible sin el config. ``val_data`` y ``train_exam_data`` son no nulas juntas,
    exactamente cuando el config declara ``data.val_root``.
    """

    sde: ForwardSDE
    model: ScoreModel
    data: Iterator
    config: TrainConfig
    model_spec: dict | None = None  # receta {name, kwargs} para el checkpoint model-agnóstico
    checkpoint: pathlib.Path | None = None
    loss_curve: pathlib.Path | None = None
    train_log: pathlib.Path | None = None  # .jsonl de estados del entrenamiento
    val_data: Iterable | None = None  # examen de validación (None = corrida sin validación)
    train_exam_data: Iterable | None = None  # examen fijo de train (None junto con val_data)


@dataclass
class DataSources:
    """Las fuentes que salen del bloque ``data:`` y la forma de evento derivada.

    Attributes:
        train: Fuente infinita de tensores crudos que consume el loop.
        event_shape: Forma de evento ``(C, H, W)`` para imágenes; ``None`` para puntos.
        val: Fuente finita y re-iterable del examen de validación, o ``None`` sin
            ``data.val_root``.
        train_exam: Examen fijo de entrenamiento, o ``None`` sin ``data.val_root``. Sale de
            ``data.root`` recortado a tantas imágenes como el set de validación, para que ambos
            exámenes tengan la misma varianza de estimador.
    """

    train: Iterator
    event_shape: tuple[int, ...] | None
    val: Iterable | None = None
    train_exam: Iterable | None = None


def load_config(path: str | pathlib.Path) -> dict:
    """Lee un archivo YAML de configuración a un ``dict``.

    Args:
        path: Ruta del ``.yaml``.

    Returns:
        El contenido parseado como ``dict``.

    Raises:
        ModuleNotFoundError: Si ``pyyaml`` no está instalado.
    """
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
        raise ModuleNotFoundError(
            "Leer configs YAML necesita PyYAML y no está instalado. "
            "Instalalo con `uv add pyyaml` (o `pip install pyyaml`)."
        ) from exc
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_data_source(data_raw: dict) -> DataSources:
    """Construye las fuentes de datos según ``data_raw['kind']`` (``points`` o ``images``).

    Para puntos envuelve la distribución de juguete en un iterador infinito. Para imágenes usa
    :func:`~diffusion.data_generation.infinite_batches` y valida la forma de evento
    ``(3, image_size, image_size)`` consumiendo el primer batch (el iterador se devuelve ya
    avanzado). Con ``val_root`` arma además las fuentes finitas de validación y del examen fijo
    de entrenamiento (desde ``root``, con tantas imágenes como ``val_root``).

    Args:
        data_raw: El bloque ``data:`` del config (no se muta). ``kind`` ausente ⇒ ``points``.
            Puntos: ``shape``/``name`` (obligatorio), ``dim`` (2), ``n_samples``,
            ``batch_size``, ``shuffle`` (True) y params de la forma. Imágenes: ``root``
            (obligatorio), ``image_size`` (64), ``batch_size``, ``augment``/``crop``/``shuffle``
            (True), ``seed`` (None), ``num_workers`` (0), ``pin_memory`` (False), ``val_root``
            (None). Otras claves se rechazan en imágenes.

    Returns:
        Un :class:`DataSources`.

    Raises:
        ValueError: Si ``kind`` no es válido, si se declara ``val_root`` en una fuente de
            puntos, si faltan claves obligatorias, si hay claves desconocidas (imágenes), si las
            carpetas no existen/están vacías/tienen menos imágenes que ``batch_size`` (solo la de
            entrenamiento), o si alguna fuente emite una forma de evento distinta.
    """
    data_raw = dict(data_raw or {})
    kind = data_raw.pop("kind", "points")
    # Se saca antes del dispatch: imágenes rechaza claves desconocidas y puntos la ignoraría.
    val_root = data_raw.pop("val_root", None)

    if kind == "points":
        if val_root is not None:
            raise ValueError(
                "config: 'data.val_root' solo aplica a kind: images; la pérdida de validación "
                "se mide sobre una fuente de imágenes y esta corrida es de puntos "
                f"(kind={kind!r}). Saca la clave o pasá la corrida a kind: images."
            )
        # El resto de las claves van a make_distribution, que filtra por firma.
        shape = data_raw.pop("shape", None) or data_raw.pop("name", None)
        if shape is None:
            raise ValueError("config: falta 'data.shape' (p. ej. gaussian / mixture / two_moons).")
        dim = data_raw.pop("dim", 2)
        n_samples = data_raw.pop("n_samples", None) or _DEFAULT_N_SAMPLES
        batch_size = data_raw.pop("batch_size", None) or _DEFAULT_BATCH_SIZE
        shuffle = data_raw.pop("shuffle", True)
        distribution = make_distribution(shape, dim, **data_raw)
        data = infinite_bare(distribution.dataloader(n_samples, batch_size, shuffle=shuffle))
        return DataSources(train=data, event_shape=None)

    if kind == "images":
        # infinite_batches no filtra kwargs por firma: las claves sobrantes se rechazan.
        root = data_raw.pop("root", None)
        if root is None:
            raise ValueError("config: falta 'data.root' para kind: images.")
        image_size = data_raw.pop("image_size", 64)
        batch_size = data_raw.pop("batch_size", None) or _DEFAULT_BATCH_SIZE
        augment = data_raw.pop("augment", True)
        crop = data_raw.pop("crop", True)
        shuffle = data_raw.pop("shuffle", True)
        seed = data_raw.pop("seed", None)
        num_workers = data_raw.pop("num_workers", 0)
        pin_memory = data_raw.pop("pin_memory", False)
        if data_raw:
            raise ValueError(
                f"config: claves desconocidas en data: para kind: images: {sorted(data_raw)}."
            )
        data = infinite_batches(
            root,
            batch_size,
            image_size=image_size,
            augment=augment,
            crop=crop,
            shuffle=shuffle,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        event_shape = (3, image_size, image_size)
        # Peek sobre el iterador real: se devuelve ya avanzado (inocuo en una fuente infinita).
        first = next(data)
        obtained = tuple(first.shape[1:])
        if obtained != event_shape:
            raise ValueError(
                f"config: la fuente de imágenes emite forma {obtained}, se esperaba "
                f"{event_shape} (derivada de image_size={image_size}, canales=3)."
            )
        # Fuentes de examen: misma forma de evento que el entrenamiento, orden canónico y sin
        # workers (una evaluación no amortiza su arranque).
        val = None
        train_exam = None
        if val_root is not None:
            val = finite_batches(
                val_root,
                batch_size,
                image_size=image_size,
                crop=crop,
                num_workers=0,
                pin_memory=False,
            )
            # Peek inocuo: la fuente es re-iterable.
            obtained_val = tuple(next(iter(val)).shape[1:])
            if obtained_val != event_shape:
                raise ValueError(
                    f"config: la fuente de validación ('data.val_root': {val_root}) emite forma "
                    f"{obtained_val}, se esperaba {event_shape} (la misma que la de "
                    "entrenamiento, para que ambas pérdidas sean comparables)."
                )
            # Examen fijo de entrenamiento: mismo tamaño que el de validación ⇒ misma varianza
            # de estimador, así el gap train↔val es atribuible al modelo.
            train_exam = finite_batches(
                root,
                batch_size,
                image_size=image_size,
                crop=crop,
                max_images=count_images(val_root),
                num_workers=0,
                pin_memory=False,
            )
        return DataSources(train=data, event_shape=event_shape, val=val, train_exam=train_exam)

    raise ValueError(
        f"config: data.kind desconocido: {kind!r}. Válidos: {', '.join(_VALID_KINDS)}."
    )


def build_run(raw: dict) -> RunSpec:
    """Ensambla un :class:`RunSpec` desde un ``dict`` de configuración.

    Args:
        raw: Config parseado (de :func:`load_config` o construido a mano).

    Returns:
        Un :class:`RunSpec` con la SDE, la red, la fuente de datos infinita, el
        :class:`TrainConfig`, las rutas y —si el config declara ``data.val_root``— las dos
        fuentes de examen: la de validación y la del examen fijo de entrenamiento.

    Raises:
        ValueError: Si faltan claves obligatorias (``sde.name``, ``data.shape``), si una corrida
            de imágenes declara ``data_dim`` en ``sde:``, si ``train:`` trae claves desconocidas,
            si ``model.name`` no está registrado, o si ``model.score_parametrization`` no es
            ``"epsilon"``. Propaga los errores de :func:`build_data_source`.
    """
    raw = dict(raw or {})

    # --- datos: va antes que la SDE porque la forma de evento derivada la configura ---
    sources = build_data_source(raw.get("data") or {})
    data, event_shape = sources.train, sources.event_shape

    # --- SDE ---
    sde_raw = dict(raw.get("sde") or {})
    if "name" not in sde_raw:
        raise ValueError("config: falta 'sde.name' (p. ej. vp / ve / sub_vp).")
    if event_shape is not None:
        # La forma se deriva de 'data:'; declararla también en 'sde:' podría desincronizarse.
        if "data_dim" in sde_raw:
            raise ValueError(
                "config: no declares 'data_dim' en sde: para imágenes; se deriva de data: "
                f"(forma de evento {event_shape})."
            )
        sde = make_sde(**sde_raw, data_dim=event_shape)
    else:
        sde = make_sde(**sde_raw)

    # --- hiperparámetros del loop -> TrainConfig (validación estricta contra sus campos) ---
    train_raw = dict(raw.get("train") or {})
    valid = {f.name for f in fields(TrainConfig)}
    unknown = set(train_raw) - valid
    if unknown:
        raise ValueError(
            f"config: claves desconocidas en train: {sorted(unknown)}. Válidas: {sorted(valid)}."
        )
    config = TrainConfig(**train_raw)

    # --- red: default 'unet' para imágenes, 'mlp' para puntos; make_model filtra por firma ---
    model_raw = dict(raw.get("model") or {})
    default_model = "unet" if event_shape is not None else "mlp"
    model_name = model_raw.pop("name", default_model)
    # No es un hiperparámetro de la red: no va al constructor ni a los kwargs de la receta.
    parametrization = model_raw.pop("score_parametrization", None)
    if parametrization is not None and parametrization != "epsilon":
        raise ValueError(
            f"config: score_parametrization desconocida: {parametrization!r}. "
            "Válidas: 'epsilon' (u omitir la clave para la red pelada)."
        )
    # data_dim solo se inyecta si es entero (MLP); la U-Net no lo toma.
    if isinstance(sde.data_dim, int):
        model_raw.setdefault("data_dim", sde.data_dim)
    model = make_model(model_name, **model_raw)
    # Receta {name, kwargs} para reconstruir la red desde el checkpoint con make_model.
    model_spec = {"name": model_name, "kwargs": dict(model_raw)}
    if parametrization == "epsilon":
        # score = -red/σ_t. La clave va junto a name/kwargs en la receta; los kwargs quedan
        # limpios para make_model y el wrap se aplica después.
        model = EpsilonScoreWrapper(model, lambda x, t: sde.marginal_prob(x, t)[1])
        model_spec["score_parametrization"] = "epsilon"

    # --- salidas ---
    out_raw = dict(raw.get("out") or {})
    checkpoint = out_raw.get("checkpoint")
    loss_curve = out_raw.get("loss_curve")
    train_log = out_raw.get("train_log")
    return RunSpec(
        sde=sde,
        model=model,
        data=data,
        config=config,
        model_spec=model_spec,
        checkpoint=pathlib.Path(checkpoint) if checkpoint else None,
        loss_curve=pathlib.Path(loss_curve) if loss_curve else None,
        train_log=pathlib.Path(train_log) if train_log else None,
        val_data=sources.val,
        train_exam_data=sources.train_exam,
    )
