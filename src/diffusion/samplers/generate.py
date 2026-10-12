"""Generación de muestras a partir de un checkpoint entrenado.

Reconstruye la red de score y la SDE desde la metadata del checkpoint, arma el sampler
pedido y opcionalmente guarda las muestras en un ``.npz``.

Uso típico::

    from diffusion.samplers import generate_from_checkpoint

    x0 = generate_from_checkpoint(
        "checkpoints/vp_mixture.pt", "pf_ode",
        n_samples=2000, n_steps=500, seed=0, out="data/vp_mixture_pf_ode.npz",
    )
"""

from __future__ import annotations

import pathlib
import warnings

import torch

from ..models import EpsilonScoreWrapper, ScoreModel, make_model
from ..sde import ForwardSDE, make_sde
from ..training import load_checkpoint


def generate_from_checkpoint(
    checkpoint_path: str | pathlib.Path,
    sampler_name: str,
    *,
    n_samples: int,
    n_steps: int = 500,
    seed: int | None = None,
    out: str | pathlib.Path | None = None,
    save_trajectory: bool = False,
    map_location: str = "cpu",
    model: ScoreModel | None = None,
    device: str | None = None,
    **sampler_kwargs,
) -> torch.Tensor:
    """Genera muestras ``x_0`` a partir de un checkpoint entrenado y opcionalmente las guarda.

    Ver :func:`load_score_model` para la reconstrucción de la red y la SDE; si la receta
    declara ``score_parametrization: "epsilon"``, el score es ``-red(x, t) / σ_t``.

    Args:
        checkpoint_path: Ruta del ``.pt`` producido por
            :func:`diffusion.training.save_checkpoint`.
        sampler_name: Clave del sampler en el registry (``"euler"``, ``"pf_ode"``,
            ``"heun"``, ``"pc"``).
        n_samples: Número de muestras a generar (``N``).
        n_steps: Número de pasos de integración del sampler.
        seed: Si no es ``None``, siembra un :class:`torch.Generator` para reproducibilidad.
        out: Ruta opcional del ``.npz`` de salida (claves ``samples`` y, si corresponde,
            ``trajectory``); crea los directorios intermedios.
        save_trajectory: Si es ``True``, guarda también la trayectoria en el ``.npz``.
        map_location: Dispositivo donde cargar los pesos del checkpoint.
        device: Device donde correr la generación; ``None`` = CPU.
        model: Red a la que cargarle los pesos; solo se usa si el checkpoint no trae la
            receta ``meta["model"]``.
        **sampler_kwargs: Parámetros extra del sampler; los no aplicables se descartan.

    Returns:
        Las muestras ``x_0`` de shape ``(n_samples, *forma_de_evento)`` en ``float32``.

    Raises:
        FileNotFoundError: Si ``checkpoint_path`` no existe.
        KeyError: Si la metadata no tiene ``sde_name``/``data_dim``.
        ValueError: Si ``sampler_name`` no está registrado, si no hay receta de red ni
            ``model=``, o si ``score_parametrization`` no es ``"epsilon"``.
    """
    # Import diferido: ``make_sampler`` vive en ``__init__``, que importa este módulo.
    from . import make_sampler

    net, sde, _ = load_score_model(
        checkpoint_path, map_location=map_location, model=model, device=device
    )

    generator: torch.Generator | None = None
    if seed is not None:
        # randn exige que el generator esté en el mismo device que el sampleo.
        generator = torch.Generator(device=device) if device is not None else torch.Generator()
        generator.manual_seed(int(seed))

    sampler = make_sampler(sampler_name, sde, net, n_steps=n_steps, **sampler_kwargs)
    result = sampler.sample(
        n_samples, generator=generator, device=device, return_trajectory=save_trajectory
    )
    if save_trajectory:
        x0, trajectory = result
    else:
        x0, trajectory = result, None

    if out is not None:
        import numpy as np

        out_path = pathlib.Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"samples": x0.cpu().numpy()}
        if save_trajectory:
            arrays["trajectory"] = trajectory.cpu().numpy()
        np.savez(out_path, **arrays)

    return x0


def load_score_model(
    checkpoint_path: str | pathlib.Path,
    *,
    map_location: str = "cpu",
    model: ScoreModel | None = None,
    device: str | None = None,
) -> tuple[torch.nn.Module, ForwardSDE, dict]:
    """Reconstruye la red de score (en ``eval``) y la SDE de un checkpoint, sin samplear.

    La red se reconstruye desde la receta ``meta["model"]`` (o desde ``model=``) y se
    envuelve con :class:`~diffusion.models.EpsilonScoreWrapper` si la receta declara
    ``score_parametrization: "epsilon"``. La SDE usa los hiperparámetros de
    ``meta["sde_kwargs"]``; un checkpoint sin esa clave cae a los defaults con un ``UserWarning``.

    Args:
        checkpoint_path: Ruta del ``.pt``.
        map_location: Dispositivo donde cargar los pesos.
        model: Red a la que cargarle los pesos si el checkpoint no trae receta.
        device: Si no es ``None``, mueve la red a ese device.

    Returns:
        ``(net, sde, meta)``: la red lista para usar como ``score_fn``, la SDE reconstruida y
        la metadata cruda del checkpoint.

    Raises:
        FileNotFoundError, KeyError, ValueError: Igual que :func:`generate_from_checkpoint`.
    """
    path = pathlib.Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint inexistente: {path}")

    state_dict, meta = load_checkpoint(path, map_location=map_location)
    try:
        sde_name = meta["sde_name"]
        data_dim = meta["data_dim"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"Checkpoint inválido en {path}: la metadata no tiene las claves esperadas "
            "('sde_name', 'data_dim'). ¿Se guardó con diffusion.training.save_checkpoint?"
        ) from exc

    # Receta {name, kwargs} vía make_model, o la instancia ``model=`` si no hay receta.
    recipe = meta.get("model") if isinstance(meta, dict) else None
    if recipe is not None:
        net = make_model(recipe["name"], **recipe["kwargs"])
        net.load_state_dict(state_dict)
    elif model is not None:
        net = model
        net.load_state_dict(state_dict)
    else:
        raise ValueError(
            f"Checkpoint en {path} sin receta de red (meta['model']) y no se pasó `model=`: "
            "no hay con qué reconstruir la red. Guardá el checkpoint con `model_spec=` "
            "(el camino config-driven lo hace) o pasá una red vía `model=`."
        )

    sde_kwargs = meta.get("sde_kwargs")
    if sde_kwargs is None:
        warnings.warn(
            f"Checkpoint en {path} sin 'sde_kwargs' (guardado antes de que se persistieran): "
            f"se reconstruye la SDE '{sde_name}' con sus parámetros por defecto. Si se entrenó con "
            "otros, el sampleo es incorrecto.",
            stacklevel=2,
        )
        sde_kwargs = {}
    sde = make_sde(sde_name, data_dim=data_dim, **sde_kwargs)

    # Parametrización ε: hay que reconstruir el mismo wrap ``-red/σ_t`` usado al entrenar.
    parametrization = recipe.get("score_parametrization") if recipe is not None else None
    if parametrization is not None and parametrization != "epsilon":
        raise ValueError(
            f"checkpoint: score_parametrization desconocida: {parametrization!r}. "
            "Válidas: 'epsilon' (u omitir la clave para la red pelada)."
        )
    if parametrization == "epsilon":
        net = EpsilonScoreWrapper(net, lambda x, t: sde.marginal_prob(x, t)[1])

    # El wrapper delega eval() a la red interna.
    net.eval()
    if device is not None:
        net = net.to(device)

    return net, sde, meta
