"""Cuadratura 2D (suma de Riemann) sobre una malla dimensionada por la densidad.

Solo depende de ``torch``. El dominio se fija por la componente más ancha
(``|μ_k| + n_sigma · sqrt(λ_max(Σ_k))``, ampliado a ``n_sigma · prior_std`` si se
declara) y el paso por la más angosta (``min_k sqrt(λ_min(Σ_k)) / points_per_sigma``),
con las covarianzas ya evaluadas en ``t``. No se usa ``σ_t``: como
``Σ_k(t) = α_t² Σ_k + σ_t² I``, el ancho tiene piso ``α_t² λ_min(Σ_k)`` y no colapsa
cuando ``t → 0``. Si el tope ``max_points`` recorta la resolución, la malla queda
marcada con ``truncated=True``; una masa lejos de uno delata una malla insuficiente.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

#: Puntos máximos por llamada a ``fn`` (la malla se recorre por bloques de filas).
_MAX_BLOCK_POINTS: int = 1 << 18

#: Tolerancia relativa con la que se exige simetría a las covarianzas.
_SYMMETRY_TOL: float = 1e-8


@dataclass(frozen=True)
class QuadratureGrid:
    """Malla de ``n_points × n_points`` nodos equiespaciados en ``[-half_width, half_width]²``.

    Attributes:
        half_width: Medio ancho del dominio en cada eje. Finito y positivo.
        n_points: Nodos por eje (>= 2).
        truncated: ``True`` si el tope de puntos limitó la resolución pedida.
    """

    half_width: float
    n_points: int
    truncated: bool = False

    def __post_init__(self) -> None:
        """Valida ``half_width`` finito y positivo y ``n_points >= 2``."""
        if not math.isfinite(self.half_width) or self.half_width <= 0.0:
            raise ValueError(
                f"half_width debe ser finito y positivo; recibido {self.half_width!r}."
            )
        if self.n_points < 2:
            raise ValueError(
                f"n_points debe ser al menos 2 para que la malla tenga paso; "
                f"recibido {self.n_points!r}."
            )

    @property
    def spacing(self) -> float:
        """Paso entre nodos consecutivos."""
        return 2.0 * self.half_width / (self.n_points - 1)


def auto_grid(
    *,
    means: torch.Tensor,
    covariances: torch.Tensor,
    prior_std: float | None = None,
    n_sigma: float = 8.0,
    points_per_sigma: float = 6.0,
    max_points: int = 4096,
) -> QuadratureGrid:
    """Dimensiona una malla 2D a partir de los parámetros de una mixtura gaussiana.

    Ver el docstring del módulo para el criterio de dominio y paso.

    Args:
        means: Medias ``(K, 2)``.
        covariances: Covarianzas ``(K, 2, 2)`` SPD, ya evaluadas en el tiempo de
            interés (``α_t² Σ_k + σ_t² I`` para la densidad perturbada).
        prior_std: Desvío de una distribución de referencia; solo amplía el dominio.
        n_sigma: Desvíos que cubre el dominio más allá de las medias.
        points_per_sigma: Nodos por desvío de la componente más angosta.
        max_points: Tope de nodos por eje; si se excede, ``truncated=True``.

    Raises:
        ValueError: Si las formas no son ``(K, 2)`` y ``(K, 2, 2)`` con el mismo ``K``, si
            hay valores no finitos, si alguna covarianza no es simétrica definida positiva,
            o si ``prior_std``, ``n_sigma``, ``points_per_sigma`` o ``max_points`` están
            fuera de rango.
    """
    if not math.isfinite(n_sigma) or n_sigma <= 0.0:
        raise ValueError(f"n_sigma debe ser finito y positivo; recibido {n_sigma!r}.")
    if not math.isfinite(points_per_sigma) or points_per_sigma <= 0.0:
        raise ValueError(
            f"points_per_sigma debe ser finito y positivo; recibido {points_per_sigma!r}."
        )
    if max_points < 2:
        raise ValueError(
            f"max_points debe ser al menos 2 para que la malla tenga paso; recibido {max_points!r}."
        )
    if prior_std is not None and (not math.isfinite(prior_std) or prior_std <= 0.0):
        raise ValueError(
            f"prior_std debe ser finito y positivo cuando se declara; recibido {prior_std!r}."
        )

    medias, covs = _validar_parametros(means, covariances)

    autovalores = torch.linalg.eigvalsh(covs)  # (K, 2), ascendente
    if bool((autovalores[:, 0] <= 0.0).any()):
        raise ValueError(
            "covariances deben ser definidas positivas; alguna tiene autovalor mínimo "
            f"no positivo ({float(autovalores[:, 0].min())!r})."
        )

    desvio_mayor = torch.sqrt(autovalores[:, 1])  # sqrt(lambda_max) por componente
    desvio_menor = torch.sqrt(autovalores[:, 0])  # sqrt(lambda_min) por componente

    radios = medias.abs().amax(dim=1) + n_sigma * desvio_mayor
    half_width = float(radios.max())
    if prior_std is not None:
        half_width = max(half_width, n_sigma * prior_std)

    paso_objetivo = float(desvio_menor.min()) / points_per_sigma
    intervalos = 2.0 * half_width / paso_objetivo if paso_objetivo > 0.0 else math.inf
    if not math.isfinite(intervalos) or intervalos > float(max_points):
        requeridos = max_points + 1
    else:
        requeridos = max(2, math.ceil(intervalos) + 1)

    truncated = requeridos > max_points
    return QuadratureGrid(
        half_width=half_width,
        n_points=max_points if truncated else requeridos,
        truncated=truncated,
    )


def integrate(
    fn: Callable[[torch.Tensor], torch.Tensor],
    grid: QuadratureGrid,
    *,
    dtype: torch.dtype = torch.float64,
) -> float:
    """Integra una función escalar sobre la malla con una suma de Riemann.

    Calcula ``sum(fn(nodos)) * spacing²``. Nodos y acumulación van en ``dtype``
    aunque ``fn`` devuelva otra precisión.

    Args:
        fn: Función ``(N, 2) -> (N,)`` a integrar.
        grid: Malla sobre la que se integra.
        dtype: Precisión de la cuadratura.

    Raises:
        ValueError: Si ``dtype`` no es de punto flotante, o si ``fn`` no devuelve un tensor
            de forma ``(N,)`` para los ``N`` puntos que recibió.
    """
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"dtype debe ser un tipo de punto flotante de torch; recibido {dtype!r}.")

    eje = torch.linspace(-grid.half_width, grid.half_width, grid.n_points, dtype=dtype)
    filas_por_bloque = max(1, _MAX_BLOCK_POINTS // grid.n_points)

    total = torch.zeros((), dtype=dtype)
    for inicio in range(0, grid.n_points, filas_por_bloque):
        filas = eje[inicio : inicio + filas_por_bloque]
        # (len(filas), n_points, 2) -> (N, 2) recorriendo primero el eje x.
        malla_y, malla_x = torch.meshgrid(filas, eje, indexing="ij")
        puntos = torch.stack((malla_x, malla_y), dim=-1).reshape(-1, 2)
        valores = fn(puntos)
        if valores.ndim != 1 or valores.shape[0] != puntos.shape[0]:
            raise ValueError(
                f"fn debe devolver un tensor de forma ({puntos.shape[0]},) para los puntos "
                f"que recibe; devolvió forma {tuple(valores.shape)}."
            )
        total = total + valores.to(dtype).sum()

    paso = torch.tensor(grid.spacing, dtype=dtype)
    return float(total * paso * paso)


def _validar_parametros(
    means: torch.Tensor, covariances: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Valida formas ``(K, 2)``/``(K, 2, 2)``, finitud y simetría; devuelve float64.

    Raises:
        ValueError: Si las formas o los valores no cumplen el contrato.
    """
    if means.ndim != 2 or means.shape[-1] != 2:
        raise ValueError(f"means debe tener forma (K, 2); recibida {tuple(means.shape)}.")
    if covariances.ndim != 3 or covariances.shape[-2:] != (2, 2):
        raise ValueError(
            f"covariances debe tener forma (K, 2, 2); recibida {tuple(covariances.shape)}."
        )
    if covariances.shape[0] != means.shape[0]:
        raise ValueError(
            f"covariances debe traer una covarianza por media; recibidas "
            f"{covariances.shape[0]} para {means.shape[0]} medias."
        )

    medias = means.detach().to(torch.float64)
    covs = covariances.detach().to(torch.float64)

    if not bool(torch.isfinite(medias).all()):
        raise ValueError("means debe tener todos sus valores finitos.")
    if not bool(torch.isfinite(covs).all()):
        raise ValueError("covariances debe tener todos sus valores finitos.")
    if not torch.allclose(covs, covs.transpose(-1, -2), rtol=_SYMMETRY_TOL, atol=1e-12):
        raise ValueError("covariances deben ser simétricas; alguna no lo es.")

    return medias, covs
