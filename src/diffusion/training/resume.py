"""Reanudación (*resume*) de una corrida de entrenamiento.

Decide si una corrida está completa (``skip``), debe reanudar desde un snapshot (``resume``) o
empieza de cero (``fresh``), y carga/valida el punto elegido. Convención de nombres: el final es
``X.pt``, los snapshots ``X_stepNNNNN.pt`` y sus sidecars ``X_stepNNNNN.resume.pt``. La política
de decisión solo consulta el filesystem y no importa torch.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # solo para anotaciones; en runtime load_resume hace import diferido del trainer
    import torch

    from .trainer import ResumeState

# Snapshot periódico ``…_stepNNNNN.pt``; anclado al final para no matchear los sidecars.
_STEP_RE = re.compile(r"_step(\d+)\.pt$")

# Sufijo del sidecar de resume hermano de un checkpoint de pesos ``X.pt`` → ``X.resume.pt``.
_RESUME_SUFFIX = ".resume.pt"


@dataclass
class ResumePlan:
    """Decisión de reanudación (DTO puro, sin persistencia).

    Attributes:
        action: ``"skip"`` (corrida ya completa), ``"fresh"`` (desde cero) o ``"resume"``
            (continuar desde ``weights_path``).
        weights_path: Checkpoint de pesos desde el que reanudar (solo en ``"resume"``).
        step: Paso ya completado del checkpoint elegido (solo en ``"resume"``; ``None`` si no se
            pudo parsear del nombre).
    """

    action: str
    weights_path: pathlib.Path | None = None
    step: int | None = None


def resume_sidecar_path(weights_path: pathlib.Path) -> pathlib.Path:
    """Devuelve la ruta del sidecar de resume: ``X_stepNNNNN.pt`` → ``X_stepNNNNN.resume.pt``."""
    return pathlib.Path(weights_path).with_suffix(_RESUME_SUFFIX)


def discover_snapshots(
    final_checkpoint: pathlib.Path,
) -> list[tuple[int, pathlib.Path]]:
    """Descubre los snapshots periódicos hermanos de un checkpoint final, ordenados por paso.

    Solo considera ``{stem}_stepNNNNN.pt`` del mismo ``stem``; excluye el final, los sidecars y
    otros archivos hermanos (p. ej. ``_best.pt``, ``_raw.pt``).

    Args:
        final_checkpoint: Ruta del checkpoint final ``X.pt`` (puede no existir).

    Returns:
        Lista de ``(step, path)`` ordenada ascendente por ``step``; vacía si no hay snapshots.
    """
    final = pathlib.Path(final_checkpoint)
    parent = final.parent
    if not parent.is_dir():
        return []

    pattern = re.compile(re.escape(final.stem) + r"_step(\d+)\.pt$")
    snaps: list[tuple[int, pathlib.Path]] = []
    for entry in parent.iterdir():
        m = pattern.fullmatch(entry.name)
        if m is not None:
            snaps.append((int(m.group(1)), entry))
    snaps.sort(key=lambda pair: pair[0])
    return snaps


def prune_snapshots(final_checkpoint: pathlib.Path, keep_last: int) -> list[pathlib.Path]:
    """Borra los snapshots intermedios más viejos, conservando los ``keep_last`` más nuevos.

    Cada snapshot se borra junto con su sidecar; el checkpoint final y su ``_raw`` nunca se
    tocan. Un sidecar ausente no es un error.

    Args:
        final_checkpoint: Ruta del checkpoint final ``X.pt`` (puede no existir todavía).
        keep_last: Cantidad de snapshots más nuevos a conservar (``>= 1``).

    Returns:
        Rutas efectivamente borradas, de más viejo a más nuevo.

    Raises:
        ValueError: Si ``keep_last < 1``.
    """
    if keep_last < 1:
        raise ValueError(
            f"keep_last debe ser >= 1 (recibido {keep_last}): conservar cero snapshots dejaría "
            "la corrida sin punto de reanudación."
        )
    snaps = discover_snapshots(final_checkpoint)  # ascendente por paso
    if len(snaps) <= keep_last:
        return []
    borrados: list[pathlib.Path] = []
    for _step, weights_path in snaps[:-keep_last]:
        for archivo in (weights_path, resume_sidecar_path(weights_path)):
            if archivo.exists():
                archivo.unlink()
                borrados.append(archivo)
    return borrados


def resolve_resume(
    final_checkpoint: pathlib.Path | None,
    *,
    force: bool = False,
    resume_from: str | None = None,
) -> ResumePlan:
    """Decide la acción de resume (``skip`` / ``fresh`` / ``resume``) a partir del config.

    Orden de resolución:

    1. ``resume_from`` (ruta o número de paso) manda: se reanuda desde ese snapshot.
    2. Si el final existe y no hay ``force`` → ``skip``.
    3. Si no, ``resume`` desde el snapshot más nuevo, o ``fresh`` si no hay ninguno (también con
       ``final_checkpoint=None``).

    No muta el filesystem.

    Args:
        final_checkpoint: Ruta del checkpoint final (``spec.checkpoint``) o ``None``.
        force: Si es ``True`` se ignora la existencia del final.
        resume_from: Ruta a un snapshot existente o número de paso de un snapshot descubierto.

    Returns:
        El :class:`ResumePlan` con la acción resuelta.

    Raises:
        ValueError: Si ``resume_from`` no resuelve a ningún snapshot; el mensaje lista los
            disponibles.
    """
    final = pathlib.Path(final_checkpoint) if final_checkpoint is not None else None

    # 1) Selección explícita: manda sobre el skip automático.
    if resume_from is not None:
        weights_path, step = _resolve_resume_from(final, resume_from)
        return ResumePlan("resume", weights_path=weights_path, step=step)

    # 2) Final ya presente (sin force) → corrida completa.
    if final is not None and final.exists() and not force:
        return ResumePlan("skip")

    # 3) Final ausente o force: reanudar desde el más nuevo si hay snapshots; si no, desde cero.
    snaps = discover_snapshots(final) if final is not None else []
    if snaps:
        step, weights_path = snaps[-1]  # el más nuevo
        return ResumePlan("resume", weights_path=weights_path, step=step)
    return ResumePlan("fresh")


def _resolve_resume_from(
    final: pathlib.Path | None, resume_from: str
) -> tuple[pathlib.Path, int | None]:
    """Resuelve ``resume_from`` a ``(weights_path, step)``: primero como ruta, luego como paso."""
    snaps = discover_snapshots(final) if final is not None else []

    # (a) ¿es una ruta a un snapshot existente?
    candidate = pathlib.Path(resume_from)
    if candidate.exists():
        return candidate, _parse_step(candidate.name)

    # (b) ¿es un número de paso que matchea un snapshot descubierto?
    if resume_from.isdigit():
        wanted = int(resume_from)
        for step, path in snaps:
            if step == wanted:
                return path, step

    raise ValueError(_unresolved_message(resume_from, snaps))


def _parse_step(name: str) -> int | None:
    """Extrae el entero del paso de un nombre ``…_stepNNNNN.pt`` (``None`` si no matchea)."""
    m = _STEP_RE.search(name)
    return int(m.group(1)) if m is not None else None


def _unresolved_message(resume_from: str, snaps: list[tuple[int, pathlib.Path]]) -> str:
    """Arma el mensaje de error de ``--resume-from`` inexistente, listando los disponibles."""
    if snaps:
        steps = ", ".join(str(step) for step, _ in snaps)
        disponibles = f"Snapshots disponibles (por paso): [{steps}]."
    else:
        disponibles = "No hay snapshots disponibles para reanudar."
    return (
        f"No se pudo resolver --resume-from={resume_from!r} "
        f"(no es una ruta existente ni un paso descubierto). {disponibles}"
    )


# ------------------------------------------- carga y validación del punto


def validate_compatible(
    meta: dict,
    *,
    sde_name: str,
    model_spec: dict | None,
    data_dim,
    sde_kwargs: dict | None = None,
) -> None:
    """Valida que el ``meta`` de un checkpoint corresponda a la corrida actual.

    Compara por igualdad exacta la SDE (nombre y, si se pasan, hiperparámetros), el ``data_dim``
    (``int`` o tupla) y la receta de red (ambas ``None`` cuenta como coincidencia). Un checkpoint
    sin ``sde_kwargs`` se interpreta como entrenado con los defaults de su SDE.

    Args:
        meta: Metadata del checkpoint de pesos (de :func:`~diffusion.training.load_checkpoint`).
        sde_name: Nombre de la SDE de la corrida actual.
        model_spec: Receta de red ``{name, kwargs}`` de la corrida actual, o ``None``.
        data_dim: ``data_dim`` de la corrida actual.
        sde_kwargs: Hiperparámetros de la SDE de la corrida actual (``sde.hparams()``), o
            ``None`` para no compararlos.

    Raises:
        ValueError: Si difiere la SDE, el ``data_dim`` o la receta de red; el mensaje detalla qué
            no coincide.
    """
    mismatches: list[str] = []
    if meta.get("sde_name") != sde_name:
        mismatches.append(f"SDE (checkpoint={meta.get('sde_name')!r} vs corrida={sde_name!r})")
    if sde_kwargs is not None and meta.get("sde_name") == sde_name:
        ckpt_kwargs = meta.get("sde_kwargs")
        if ckpt_kwargs is None:
            # Import diferido: el módulo no arrastra torch al importarse.
            from ..sde import make_sde

            ckpt_kwargs = make_sde(sde_name).hparams()
        if ckpt_kwargs != sde_kwargs:
            mismatches.append(
                f"hiperparámetros de la SDE (checkpoint={ckpt_kwargs!r} vs corrida={sde_kwargs!r})"
            )
    if meta.get("data_dim") != data_dim:
        mismatches.append(f"data_dim (checkpoint={meta.get('data_dim')!r} vs corrida={data_dim!r})")
    if meta.get("model") != model_spec:
        mismatches.append(
            f"receta de red (checkpoint={meta.get('model')!r} vs corrida={model_spec!r})"
        )
    if mismatches:
        raise ValueError(
            "El checkpoint elegido es incompatible con la corrida actual y no se puede reanudar "
            "desde un estado que no corresponde. Difiere: " + "; ".join(mismatches) + "."
        )


def load_resume(
    weights_path: pathlib.Path,
    *,
    expected: dict,
    map_location: torch.device | str = "cpu",
) -> tuple[dict, dict, ResumeState]:
    """Carga un punto de reanudación (pesos + sidecar) y arma el :class:`ResumeState`.

    Valida compatibilidad antes de exigir el sidecar. El optimizador, el paso, el azar, la sombra
    EMA y el escalador AMP salen del sidecar; ``history`` y ``val_history`` del ``meta`` del
    checkpoint de pesos. Con EMA el checkpoint publica la sombra, así que el ``state_dict`` a
    cargar en la red son los pesos crudos del sidecar (``raw_model_state``) cuando están.

    Args:
        weights_path: Ruta del checkpoint de pesos elegido (``…_stepNNNNN.pt``).
        expected: ``{"sde_name", "model_spec", "data_dim"}`` (y opcionalmente ``"sde_kwargs"``)
            de la corrida actual (se pasa a :func:`validate_compatible`).
        map_location: Dispositivo donde cargar los tensores.

    Returns:
        ``(state_dict, meta, resume)``: los pesos a cargar en la red, el ``meta`` del checkpoint y
        el :class:`ResumeState` listo para ``train(resume=...)``.

    Raises:
        ValueError: Si el checkpoint es incompatible con la corrida.
        FileNotFoundError: Si falta el sidecar de resume del checkpoint elegido.
    """
    # Import diferido: el módulo no arrastra torch al importarse.
    from .trainer import ResumeState, load_checkpoint, load_resume_state

    weights_path = pathlib.Path(weights_path)

    # 1) pesos + meta del checkpoint elegido.
    state_dict, meta = load_checkpoint(weights_path, map_location=map_location)

    # 2) compatibilidad, antes de exigir el sidecar.
    validate_compatible(meta, **expected)

    # 3) el sidecar es obligatorio.
    sidecar = resume_sidecar_path(weights_path)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"Falta el sidecar de resume esperado: {sidecar}. Sin el estado del "
            f"optimizador/paso/azar no se puede reanudar desde {weights_path.name} "
            "(¿la corrida original corrió con checkpoint_every>0?)."
        )

    # 4) estado del sidecar → ResumeState.
    sc = load_resume_state(sidecar, map_location=map_location)

    # Con EMA el checkpoint publica la sombra: hay que continuar desde los crudos del sidecar.
    crudos = sc.get("raw_model_state")
    if crudos is not None:
        state_dict = crudos

    resume = ResumeState(
        optimizer_state=sc["optimizer_state"],
        start_step=sc["step"],
        torch_rng_state=sc["torch_rng_state"],
        generator_state=sc["generator_state"],
        history=meta["history"],  # del checkpoint de pesos, no del sidecar
        ema_state=sc.get("ema_state"),
        scaler_state=sc.get("scaler_state"),
        val_history=meta.get("val_history"),  # ausente en corridas sin validación
    )
    return state_dict, meta, resume
