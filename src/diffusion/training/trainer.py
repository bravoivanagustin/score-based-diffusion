"""Loop de entrenamiento por DSM y persistencia de checkpoints.

:func:`train` es agnóstico a la red y al origen de datos: recibe la red ya construida y un
iterador infinito de tensores, y corre ``config.num_steps`` pasos. :func:`save_checkpoint` /
:func:`load_checkpoint` guardan el ``state_dict`` con una metadata mínima (SDE, ``data_dim`` y una
receta ``model={name, kwargs}`` opcional) sin reconstruir la red; eso queda en manos del caller.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field

import torch

from ..models import ScoreModel
from ..sde import ForwardSDE
from .ema import EmaShadow
from .losses import dsm_loss
from .time_sampling import make_time_sampler
from .validation import FixedValExam, ValPoint, evaluate_with_weights


def _enable_cudnn_autotune(device: torch.device) -> None:
    """Activa ``cudnn.benchmark`` en CUDA (los shapes son fijos); en CPU no hace nada."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


@dataclass
class TrainConfig:
    """Hiperparámetros del loop de entrenamiento (optimización y corrida).

    Los hiperparámetros de red viven en ``make_model`` y los de datos en la fuente de datos.
    """

    num_steps: int = 1000  # pasos de optimización
    lr: float = 2e-3
    t_eps: float = 1e-3
    grad_clip: float | None = None
    seed: int | None = 0
    device: str = "cpu"
    log_every: int = 0  # 0 = silencioso; N = imprime la media móvil cada N pasos (solo consola)
    # 0 = solo el checkpoint final; N>0 = además un snapshot cada N pasos (vía ``on_checkpoint``).
    checkpoint_every: int = 0
    # None = conservar todos los snapshots; N >= 1 = conservar los N más nuevos. Lo aplica el CLI
    # con :func:`~diffusion.training.prune_snapshots`; ``train`` no lo usa.
    keep_last_checkpoints: int | None = None
    # Cadencia de la validación: None = usar ``checkpoint_every``; N >= 1 = cada N pasos,
    # independiente de los snapshots (conviene que ``checkpoint_every`` sea múltiplo).
    val_every: int | None = None
    # Distribución de t (ver :mod:`diffusion.training.time_sampling`): "uniform" o "log_uniform".
    time_sampling: str = "uniform"
    # Decay de la sombra EMA en (0, 1), p. ej. 0.999; None = sin EMA. Los checkpoints la publican.
    ema_decay: float | None = None
    # Precisión mixta (bool): autocast en forward/pérdida y GradScaler en GPU (bfloat16 en CPU).
    amp: bool = False


@dataclass
class TrainResult:
    """Resultado de :func:`train`: la red entrenada y la traza de la corrida."""

    net: ScoreModel
    history: list[float] = field(default_factory=list)  # pérdida de cada paso
    config: TrainConfig = field(default_factory=TrainConfig)
    sde_name: str = ""
    # Hiperparámetros de la SDE (``sde.hparams()``): con ``sde_name`` y ``data_dim`` la reconstruyen.
    # None = desconocidos (TrainResult armado a mano); el checkpoint omite entonces la clave.
    sde_kwargs: dict | None = None
    # sde.data_dim: entero para dato plano o tupla (forma de evento) para imágenes.
    data_dim: int | tuple[int, ...] = 0
    # Foto clonada de la sombra EMA (claves del state_dict de la red); None sin EMA.
    ema_state: dict | None = None
    # Serie dispersa de validación (un ValPoint por evaluación); vacía sin fuente de validación.
    val_history: list[ValPoint] = field(default_factory=list)


@dataclass
class ResumeState:
    """Estado para reanudar una corrida: lo que no se recupera de los pesos.

    ``history`` y ``val_history`` viajan en memoria pero no se persisten en el sidecar: viven en el
    ``meta`` del checkpoint de pesos. Los opcionales quedan en ``None`` sin EMA/AMP/validación.

    Attributes:
        optimizer_state: ``optimizer.state_dict()``.
        start_step: Pasos ya completados (= ``N`` del nombre ``…_stepNNNNN``).
        torch_rng_state: ``torch.get_rng_state()``.
        generator_state: ``generator.get_state()`` (ruido del kernel / muestreo de ``t``).
        history: Pérdida per-step hasta ``start_step``.
        ema_state: Foto clonada de la sombra EMA, o ``None``. Se restaura con
            ``EmaShadow.load_state`` (una sombra nueva perdería el promedio acumulado).
        raw_model_state: Pesos crudos clonados, o ``None``. Con EMA el checkpoint publica la
            sombra, así que estos son los pesos desde los que se continúa optimizando.
        scaler_state: ``GradScaler.state_dict()``, o ``None`` sin AMP. En CPU es ``{}``: la
            presencia se decide por ``is None``.
        val_history: Serie de validación hasta ``start_step``, o ``None``.
    """

    optimizer_state: dict
    start_step: int
    torch_rng_state: torch.Tensor
    generator_state: torch.Tensor
    history: list[float]
    ema_state: dict | None = None
    raw_model_state: dict | None = None
    scaler_state: dict | None = None
    val_history: list[ValPoint] | None = None


@dataclass
class TrainSnapshot:
    """Estado de un checkpoint intermedio: lo que persisten el checkpoint de pesos y el sidecar.

    Attributes:
        result: Pesos + history (para :func:`save_checkpoint`).
        resume: Estado para reanudar (para :func:`save_resume_state`).
    """

    result: TrainResult
    resume: ResumeState


def train(
    sde: ForwardSDE,
    model: ScoreModel,
    data: Iterator[torch.Tensor],
    config: TrainConfig,
    *,
    generator: torch.Generator | None = None,
    on_checkpoint: Callable[[str, TrainSnapshot], None] | None = None,
    on_log: Callable[[dict], None] | None = None,
    resume: ResumeState | None = None,
    progress: bool = False,
    val_batches: Iterable[torch.Tensor] | None = None,
    train_exam_batches: Iterable[torch.Tensor] | None = None,
) -> TrainResult:
    """Entrena la red ``model`` para aproximar el score de ``sde`` por DSM.

    Consume un batch de ``data`` por paso. Sin ``resume`` entrena desde cero sembrando con
    ``config.seed``; con ``resume`` restaura optimizador y azar sin re-sembrar y continúa hasta
    ``config.num_steps`` (el total, no pasos adicionales).

    Con ``config.ema_decay`` mantiene una sombra EMA pasiva: con la misma semilla, la trayectoria
    de optimización es idéntica a la de una corrida sin EMA. Con ``config.amp`` el forward y la
    pérdida corren bajo ``torch.autocast`` y, en GPU, el gradiente se escala con ``GradScaler``
    (se des-escala antes de ``grad_clip``). Con ``val_batches`` mide la pérdida de validación por
    examen fijo cada ``val_every`` (o ``checkpoint_every``) pasos y en el último, después de
    ``ema.update`` y antes del snapshot, así el checkpoint del paso N ya incluye el punto de N.

    Args:
        sde: Proceso forward (define el kernel y el target del score).
        model: Red de score ya construida. Se mueve al dispositivo y se pone en modo
            entrenamiento. Al reanudar, el caller ya le cargó los pesos.
        data: Iterador infinito de tensores crudos ``(B, ...)``.
        config: Hiperparámetros del loop.
        generator: Generador para el ruido del kernel / muestreo de ``t``. Si es ``None`` se crea;
            sin ``resume`` se siembra con ``config.seed``, con ``resume`` se restaura su estado.
        on_checkpoint: Callback ``(tag, snapshot)`` invocado cada ``config.checkpoint_every``
            pasos (excepto el último) con ``tag = "step{N:05d}"`` y un :class:`TrainSnapshot`.
            ``train`` no toca el filesystem: el callback decide cómo persistir.
        on_log: Callback de logging estructurado con dos variantes de registro:

            - Paso: ``{"step", "loss"}`` (media móvil), sin clave ``event``, cada
              ``config.log_every`` pasos o ``num_steps//20`` si es 0.
            - Validación: ``{"event": "val", "step", "val_raw", "val_ema", "train_fijo",
              "device"}`` en cada evaluación. El ``device`` va porque el examen fijo es
              específico del device.
        resume: Estado para reanudar. Si ``start_step >= num_steps`` no corre ningún paso.
        progress: Si es ``True`` muestra una barra ``tqdm`` (solo display).
        val_batches: Fuente re-iterable de batches de validación. Requiere una cadencia efectiva
            > 0. La validación no altera pesos, optimizador, sombra EMA ni RNG.
        train_exam_batches: Fuente re-iterable del examen fijo de entrenamiento (mismo tamaño
            que el de validación). Se ignora sin ``val_batches``.

    Returns:
        :class:`TrainResult` con la red, la pérdida de cada paso, la config, el nombre y el
        ``data_dim`` de la SDE, la sombra EMA (o ``None``) y la serie de validación.

    Raises:
        ValueError: Si ``config.ema_decay`` es inválido, si ``config.amp`` no es ``bool``, si
            ``config.time_sampling`` no existe, si ``config.val_every < 1``, si hay
            ``val_batches`` sin cadencia efectiva, o si al reanudar la config y el sidecar no
            coinciden en EMA o AMP. Todos antes de consumir el primer batch.
    """
    device = torch.device(config.device)
    _enable_cudnn_autotune(device)
    if resume is None:
        if config.seed is not None:
            torch.manual_seed(config.seed)
        if generator is None:
            generator = torch.Generator(device=device)
            if config.seed is not None:
                generator.manual_seed(config.seed)
    elif generator is None:
        # Reanudación: no se re-siembra; el estado se restaura más abajo.
        generator = torch.Generator(device=device)

    # Fail-fast ante un nombre desconocido o t_eps fuera de (0, T); no consume RNG.
    time_sampler = make_time_sampler(config.time_sampling, sde.T, config.t_eps)

    net = model.to(device)
    net.train()

    # Después de mover la red (la sombra clona sus tensores) y antes de consumir datos.
    ema = EmaShadow(net, config.ema_decay) if config.ema_decay is not None else None

    # bool estricto: 0/1 no cuentan.
    if not isinstance(config.amp, bool):
        raise ValueError(
            f"config.amp debe ser bool (True/False); recibido {config.amp!r} "
            f"(tipo {type(config.amp).__name__})."
        )
    # Solo con AMP; en CPU existe pero deshabilitado (bfloat16 no necesita escalado).
    scaler = (
        torch.amp.GradScaler(device.type, enabled=(device.type == "cuda")) if config.amp else None
    )

    # La cadencia se valida acá y no en la config porque el CLI puede sobreescribir
    # ``checkpoint_every`` después de armar la corrida.
    val_cadence = config.val_every if config.val_every is not None else config.checkpoint_every
    # El valor declarado se valida siempre, haya o no fuente de validación.
    if config.val_every is not None and config.val_every < 1:
        raise ValueError(
            f"config.val_every={config.val_every} no es una cadencia válida: la validación se "
            "evalúa cada N pasos, así que N tiene que ser >= 1. Declarar 0 o un negativo NO es "
            "'usar el default' —eso lo dice val_every: null, que cae en checkpoint_every—: es "
            "pedir una cadencia imposible, y se rechaza en vez de degradar en silencio."
        )
    val_exam: FixedValExam | None = None
    train_exam: FixedValExam | None = None
    if val_batches is not None:
        if val_cadence <= 0:
            raise ValueError(
                "La corrida pide pérdida de validación pero se queda sin cadencia de evaluación: no "
                f"declara val_every y tiene los checkpoints deshabilitados "
                f"(config.checkpoint_every={config.checkpoint_every}), así que entrenaría completa "
                "sin medir ni un punto. Declará val_every > 0 —fija la cadencia de la validación con "
                "independencia de los snapshots, así que podés medir sin escribir ninguno— o poné "
                "checkpoint_every > 0; si no querés medir, quitá la fuente de validación."
            )
        # Mismo time_sampler que el loop y misma semilla: ambos exámenes sortean la misma secuencia.
        val_exam = FixedValExam(sde, val_batches, time_sampler=time_sampler, device=device)
        if train_exam_batches is not None:
            train_exam = FixedValExam(
                sde, train_exam_batches, time_sampler=time_sampler, device=device
            )

    # Coherencia config ↔ sidecar al reanudar: se rechazan las combinaciones cruzadas de EMA y AMP.
    if resume is not None:
        if ema is not None and resume.ema_state is None:
            raise ValueError(
                f"La corrida configuró ema_decay={config.ema_decay} pero el punto de reanudación "
                "no trae la sombra EMA: el sidecar de resume del checkpoint elegido no la "
                "persistió (¿la corrida original entrenó sin EMA?). No se reanuda con una sombra "
                "inventada —perdería el promedio de todos los pasos ya corridos y el checkpoint "
                "final publicaría un EMA que no corresponde a la corrida—: entrená desde cero o "
                "reanudá con la misma configuración que la corrida original."
            )
        if ema is None and resume.ema_state is not None:
            raise ValueError(
                "El punto de reanudación trae una sombra EMA en su sidecar de resume (la corrida "
                "original la mantenía) pero esta corrida no configuró ema_decay: continuar "
                "descartaría la sombra en silencio y el checkpoint final publicaría los pesos "
                "crudos donde los intermedios de la misma corrida publicaron el promedio. "
                "Declará el ema_decay de la corrida original o reanudá desde otro punto."
            )
        if ema is not None:
            # La sombra recién construida vale θ_N, no el promedio acumulado.
            ema.load_state(resume.ema_state)

        if scaler is not None and resume.scaler_state is None:
            raise ValueError(
                "La corrida configuró amp=True pero el punto de reanudación no trae el estado del "
                "escalador de gradiente: el sidecar de resume del checkpoint elegido no lo "
                "persistió (¿la corrida original entrenó sin AMP?). No se reanuda con un escalador "
                "inventado —el factor de escala continuaría desde un valor que no corresponde a la "
                "corrida—: entrená desde cero o reanudá con la misma configuración de precisión que "
                "la corrida original."
            )
        if scaler is None and resume.scaler_state is not None:
            raise ValueError(
                "El punto de reanudación trae el estado del escalador de gradiente en su sidecar de "
                "resume (la corrida original usaba amp=True) pero esta corrida no configuró amp: "
                "continuar descartaría el escalador en silencio y la reanudación no sería fiel a la "
                "corrida original. Declará amp=True como en la corrida original o reanudá desde otro "
                "punto."
            )
        if scaler is not None:
            # El escalador nuevo arrancaría con el factor de escala inicial.
            scaler.load_state_dict(resume.scaler_state)

    data_iter = iter(data)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    if resume is None:
        start_step = 0
        history: list[float] = []
        val_history: list[ValPoint] = []
    else:
        # load_state_dict levanta además si las shapes del optimizador no corresponden.
        optimizer.load_state_dict(resume.optimizer_state)
        torch.set_rng_state(resume.torch_rng_state)
        generator.set_state(resume.generator_state)
        start_step = resume.start_step
        history = list(resume.history)
        val_history = list(resume.val_history or [])

    do_checkpoints = on_checkpoint is not None and config.checkpoint_every > 0

    def _result() -> TrainResult:
        return TrainResult(
            net=net,
            history=list(history),
            config=config,
            sde_name=sde.name,
            sde_kwargs=sde.hparams(),
            data_dim=sde.data_dim,
            ema_state=ema.state_dict() if ema is not None else None,
            val_history=list(val_history),
        )

    def _snapshot(completed_steps: int) -> TrainSnapshot:
        # El azar se lee después del paso: reanudar desde acá continúa el mismo stream. Los
        # crudos se clonan porque el loop sigue mutando la red in-place.
        return TrainSnapshot(
            result=_result(),
            resume=ResumeState(
                optimizer_state=optimizer.state_dict(),
                start_step=completed_steps,
                torch_rng_state=torch.get_rng_state(),
                generator_state=generator.get_state(),
                history=list(history),
                ema_state=ema.state_dict() if ema is not None else None,
                raw_model_state=(
                    {k: v.detach().clone() for k, v in net.state_dict().items()}
                    if ema is not None
                    else None
                ),
                scaler_state=scaler.state_dict() if scaler is not None else None,
                val_history=list(val_history),
            ),
        )

    # num_steps es el total: al reanudar solo se corren los pasos restantes.
    step_iter = range(start_step, config.num_steps)
    pbar = None
    if progress:
        # initial/total dejan el % y el ETA correctos al reanudar.
        from tqdm.auto import tqdm

        pbar = tqdm(
            step_iter,
            initial=start_step,
            total=config.num_steps,
            unit="paso",
            desc=sde.name,
            dynamic_ncols=True,
        )
        step_iter = pbar

    # Independiente del print: on_log emite estados incluso con log_every=0.
    log_cadence = config.log_every if config.log_every > 0 else max(1, config.num_steps // 20)

    for step in step_iter:
        # non_blocking es inocuo sin pin_memory o en CPU.
        x0 = next(data_iter).to(device, non_blocking=True)
        t, sample_weights = time_sampler.sample(x0.shape[0], generator=generator, device=device)
        # El backward queda fuera del autocast (contrato de torch.amp).
        with torch.autocast(device_type=device.type, enabled=config.amp):
            loss = dsm_loss(net, sde, x0, t, generator=generator, sample_weights=sample_weights)

        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            optimizer.step()
        else:
            # Des-escalar antes del clip para que opere sobre la norma real del gradiente.
            scaler.scale(loss).backward()
            if config.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        # Una vez por paso con el contador 1-indexado: la rampa de warmup depende de él.
        if ema is not None:
            ema.update(step + 1)

        history.append(loss.item())
        is_last = step == config.num_steps - 1

        # Después de ema.update y antes del snapshot: el checkpoint del paso N incluye el punto
        # de N. A diferencia del snapshot, incluye el último paso y no requiere on_checkpoint.
        if val_exam is not None and ((step + 1) % val_cadence == 0 or is_last):
            val_raw = val_exam.evaluate(net)
            # Sin EMA queda None en lugar de repetir el valor de los crudos.
            val_ema = (
                evaluate_with_weights(val_exam, net, ema.state_dict()) if ema is not None else None
            )
            # Solo con los pesos vivos: la comparación raw↔EMA se hace sobre validación.
            train_fijo = train_exam.evaluate(net) if train_exam is not None else None

            punto: ValPoint = {
                "step": step + 1,  # 1-indexado
                "raw": val_raw,
                "ema": val_ema,
                "train": train_fijo,
            }
            val_history.append(punto)
            if on_log is not None:
                on_log(
                    {
                        "event": "val",
                        "step": step + 1,
                        "val_raw": val_raw,
                        "val_ema": val_ema,
                        "train_fijo": train_fijo,
                        "device": str(device),
                    }
                )

        # El último paso lo cubre el checkpoint final del caller.
        if do_checkpoints and not is_last and (step + 1) % config.checkpoint_every == 0:
            on_checkpoint(f"step{step + 1:05d}", _snapshot(step + 1))

        if on_log is not None and ((step + 1) % log_cadence == 0 or is_last):
            recent_log = history[-log_cadence:]
            on_log({"step": step + 1, "loss": sum(recent_log) / len(recent_log)})

        # Con barra activa va como postfix para no romper su línea.
        if config.log_every > 0 and ((step + 1) % config.log_every == 0 or is_last):
            recent = history[-config.log_every :]
            avg = sum(recent) / len(recent)
            if pbar is not None:
                pbar.set_postfix(perdida=f"{avg:.6f}")
            else:
                print(f"[{sde.name}] paso {step + 1}/{config.num_steps}  pérdida(móvil)={avg:.6f}")

    if pbar is not None:
        pbar.close()

    return TrainResult(
        net=net,
        history=history,
        config=config,
        sde_name=sde.name,
        sde_kwargs=sde.hparams(),
        data_dim=sde.data_dim,
        ema_state=ema.state_dict() if ema is not None else None,
        val_history=val_history,
    )


# ----------------------------------------------------------------- persistencia


def save_checkpoint(
    result: TrainResult,
    path: str | pathlib.Path,
    *,
    model_spec: dict | None = None,
    run_config: dict | None = None,
    raw_sibling: bool = False,
) -> pathlib.Path:
    """Guarda la red entrenada y su metadata en ``path`` (``.pt`` de torch).

    Si ``result.ema_state`` está presente se publica la sombra EMA como ``model_state`` y la meta
    gana ``ema = {"decay": d}``; el formato ``{model_state, meta}`` no cambia. Convención de la meta:

    - sin ``ema`` ⇒ pesos crudos.
    - ``ema = {"decay": d}`` ⇒ sombra EMA con ese decay.
    - ``raw_of = "<archivo>.pt"`` ⇒ hermano de crudos del checkpoint EMA nombrado.
    - ``val_history`` (solo si no está vacía) ⇒ serie de validación de la corrida, como ``dict``
      planos; de ahí la recupera :func:`~diffusion.training.load_resume`.
    - ``sde_kwargs`` ⇒ hiperparámetros de la SDE; ``train_config`` ⇒ el :class:`TrainConfig`
      efectivo (con los overrides de CLI); ``run_config`` ⇒ el config crudo de la corrida, si se pasa.

    Args:
        result: Resultado de :func:`train`.
        path: Ruta de salida (se crean los directorios intermedios).
        model_spec: Receta ``{"name": str, "kwargs": dict}`` para reconstruir la red con
            :func:`diffusion.models.make_model`. Si es ``None`` no se guarda la clave ``model``.
        run_config: Config crudo de la corrida (el ``dict`` del YAML), para reproducirla. Si es
            ``None`` no se guarda la clave ``run_config``.
        raw_sibling: Si es ``True`` y la corrida publica EMA, escribe además ``{stem}_raw.pt`` con
            los pesos crudos finales y ``raw_of`` en la meta, para comparar crudo vs EMA. El
            sufijo ``_raw`` no matchea el patrón de snapshots, así que nunca se elige para reanudar.

    Returns:
        La ruta del checkpoint principal; la del hermano de crudos es
        ``out.with_stem(out.stem + "_raw")``.
    """
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "sde_name": result.sde_name,
        "data_dim": result.data_dim,
        "train_config": asdict(result.config),
        "history": list(result.history),
    }
    if model_spec is not None:
        meta["model"] = model_spec
    if result.sde_kwargs is not None:
        meta["sde_kwargs"] = dict(result.sde_kwargs)
    if run_config is not None:
        meta["run_config"] = run_config

    # Clave opcional: sin validación la meta no gana la clave.
    if result.val_history:
        meta["val_history"] = list(result.val_history)

    publica_ema = result.ema_state is not None
    if publica_ema:
        meta["ema"] = {"decay": result.config.ema_decay}
    blob = {
        "model_state": result.ema_state if publica_ema else result.net.state_dict(),
        "meta": meta,
    }
    torch.save(blob, out)

    if raw_sibling and publica_ema:
        # Misma meta sin ``ema`` y con ``raw_of``; val_history se conserva (describe la corrida).
        raw_meta = {clave: valor for clave, valor in meta.items() if clave != "ema"}
        raw_meta["raw_of"] = out.name
        torch.save(
            {"model_state": result.net.state_dict(), "meta": raw_meta},
            out.with_stem(f"{out.stem}_raw"),
        )
    return out


def load_checkpoint(
    path: str | pathlib.Path, *, map_location: torch.device | str = "cpu"
) -> tuple[dict, dict]:
    """Carga un checkpoint de :func:`save_checkpoint` como ``(state_dict, meta)``.

    No reconstruye la red: el caller la arma (p. ej. con :func:`diffusion.models.make_model` y
    ``meta["model"]``) y le carga el ``state_dict``.

    Args:
        path: Ruta del ``.pt`` guardado.
        map_location: Dispositivo donde cargar los pesos (default ``"cpu"``).

    Returns:
        ``(state_dict, meta)``: el ``state_dict`` de la red y el ``dict`` de metadata guardado.

    Raises:
        KeyError: Si el archivo no tiene la forma de un checkpoint de :func:`save_checkpoint`
            (faltan ``"model_state"`` o ``"meta"``).
    """
    # weights_only=False: es nuestro propio checkpoint (incluye un dict de metadata).
    blob = torch.load(path, map_location=map_location, weights_only=False)
    return blob["model_state"], blob["meta"]


# --------------------------------------------------- sidecar de resume

# Campos obligatorios del sidecar; ``history`` vive en el ``meta`` del checkpoint de pesos.
_REQUIRED_RESUME_FIELDS = ("optimizer_state", "start_step", "torch_rng_state", "generator_state")


def save_resume_state(path: str | pathlib.Path, resume: ResumeState) -> pathlib.Path:
    """Persiste el sidecar de resume de un checkpoint (sin ``history``).

    Guarda lo que el checkpoint de pesos no tiene: optimizador, paso y azar. Las claves opcionales
    ``raw_model_state``/``ema_state`` (EMA) y ``scaler_state`` (AMP) se escriben solo si están.

    Args:
        path: Ruta de salida (``X_stepNNNNN.resume.pt``); se crean los directorios intermedios.
        resume: Estado a persistir.

    Returns:
        La ruta donde se guardó.

    Raises:
        ValueError: Si falta alguno de ``optimizer_state`` / ``start_step`` / ``torch_rng_state``
            / ``generator_state``. No se escribe un sidecar parcial.
    """
    missing = [name for name in _REQUIRED_RESUME_FIELDS if getattr(resume, name) is None]
    if missing:
        raise ValueError(
            "Estado de resume incompleto: falta(n) "
            f"{', '.join(missing)}. No se persiste un sidecar parcial "
            "(se requieren optimizer_state, start_step, torch_rng_state y generator_state)."
        )

    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "optimizer_state": resume.optimizer_state,
        "step": resume.start_step,
        "torch_rng_state": resume.torch_rng_state,
        "generator_state": resume.generator_state,
    }
    if resume.raw_model_state is not None:
        blob["raw_model_state"] = resume.raw_model_state
    if resume.ema_state is not None:
        blob["ema_state"] = resume.ema_state
    if resume.scaler_state is not None:
        blob["scaler_state"] = resume.scaler_state
    torch.save(blob, out)
    return out


def load_resume_state(
    path: str | pathlib.Path, *, map_location: torch.device | str = "cpu"
) -> dict:
    """Carga un sidecar de :func:`save_resume_state` como ``dict``.

    Args:
        path: Ruta del sidecar ``…_resume.pt``.
        map_location: Dispositivo donde cargar los tensores (default ``"cpu"``).

    Returns:
        ``{optimizer_state, step, torch_rng_state, generator_state}`` más, si están,
        ``raw_model_state``, ``ema_state`` y ``scaler_state`` (pedirlas con ``get``).
    """
    # weights_only=False: es nuestro propio archivo (incluye el state_dict del optimizador).
    return torch.load(path, map_location=map_location, weights_only=False)
