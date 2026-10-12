"""Formas concretas de distribuciones de puntos.

Se apoya en scikit-learn donde existe un generador adecuado (``make_moons``,
``make_blobs``, ``make_swiss_roll``) y en numpy para la espiral (sklearn no trae
generador de espiral).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll

from .base import PointDistribution


def _seed_from(rng: np.random.Generator) -> int:
    """Deriva un ``random_state`` int para sklearn, atado al rng reproducible."""
    return int(rng.integers(0, 2**31 - 1))


def _split_counts(n: int, parts: int) -> list[int]:
    """Reparte ``n`` en ``parts`` enteros lo más parejos posible (suman ``n``)."""
    base, rem = divmod(n, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _largest_remainder_counts(weights: np.ndarray, n: int) -> np.ndarray:
    """Reparte ``n`` puntos entre componentes por mayor residuo (determinístico).

    Cada componente recibe ``floor(w_k * n)`` y las unidades faltantes van a los
    mayores residuos, así la suma es exactamente ``n``.
    """
    exact = weights * n
    counts = np.floor(exact).astype(np.int64)
    missing = int(n - counts.sum())
    if missing > 0:
        # ``-residues`` con orden estable => desempate por índice de componente.
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:missing]] += 1
    return counts


class Gaussian(PointDistribution):
    """Gaussiana isotrópica N-dim centrada en el origen."""

    name = "gaussian"
    supported_dims = None  # cualquier dim >= 1

    def __init__(self, dim, *, scale=1.0, standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=0.0, seed=seed)
        self.scale = float(scale)

    def _sample_raw(self, n, rng):
        return rng.normal(0.0, self.scale, size=(n, self.dim))


class GaussianMixture(PointDistribution):
    """Mezcla de gaussianas isotrópicas.

    En 2D los centros se ubican en un anillo (el clásico "8 gaussianas"); en
    otras dimensiones, en direcciones aleatorias sobre una hiperesfera de radio
    ``radius``.
    """

    name = "mixture"
    supported_dims = None

    def __init__(
        self, dim, *, n_components=8, cluster_std=0.3, radius=5.0, standardize=False, seed=None
    ):
        super().__init__(dim, standardize=standardize, noise=0.0, seed=seed)
        if n_components < 1:
            raise ValueError(f"n_components debe ser >= 1; recibí {n_components}")
        self.n_components = int(n_components)
        self.cluster_std = float(cluster_std)
        self.radius = float(radius)

    def _centers(self, rng):
        k = self.n_components
        if self.dim == 2:
            ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
            return self.radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        dirs = rng.normal(size=(k, self.dim))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        return self.radius * dirs

    def _sample_raw(self, n, rng):
        centers = self._centers(rng)
        x, y = make_blobs(
            n_samples=n,
            centers=centers,
            cluster_std=self.cluster_std,
            random_state=_seed_from(rng),
        )
        self.color_ = y
        return x


def _validate_scale_and_anisotropy(scale: float, anisotropy: float) -> None:
    """Valida ``scale > 0`` y ``anisotropy >= 1`` (rechaza también NaN)."""
    if not float(scale) > 0.0:
        raise ValueError(f"scale debe ser > 0; recibí {scale!r}")
    if not float(anisotropy) >= 1.0:
        raise ValueError(
            f"anisotropy debe ser >= 1: es la razón entre el autovalor mayor y "
            f"el menor de la componente, así que 1 es el caso isotrópico y "
            f"valores mayores la estiran; recibí {anisotropy!r}"
        )


def _anisotropic_covariance(scale: float, anisotropy: float, angle: float) -> np.ndarray:
    """Covarianza 2x2 SPD con anisotropía ``anisotropy`` y eje mayor en ``angle``.

    Los autovalores son ``scale² · √κ`` a lo largo de ``angle`` y ``scale² / √κ``
    en la perpendicular (``κ = anisotropy``): su razón es ``κ`` y su media
    geométrica es ``scale²``, así que variar ``κ`` estira la componente sin
    cambiar el determinante.

    Args:
        scale: Escala global de la componente (> 0).
        anisotropy: Razón mayor/menor de autovalores (>= 1); 1 es isotrópica.
        angle: Ángulo en radianes del eje mayor.
    """
    root = np.sqrt(float(anisotropy))
    var = float(scale) ** 2
    major = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    minor = np.array([-np.sin(angle), np.cos(angle)], dtype=np.float64)
    return var * root * np.outer(major, major) + (var / root) * np.outer(minor, minor)


class ExactGaussianMixture(PointDistribution):
    """Mixtura de gaussianas 2D con parámetros exactos y consultables.

    Pesos, medias y covarianzas (SPD arbitrarias, no solo diagonales) son
    entrada explícita y quedan expuestos en :attr:`weights_`, :attr:`means_` y
    :attr:`covariances_`, así que la densidad se conoce en forma cerrada:
    ``p(x) = Σ_k w_k N(x; μ_k, Σ_k)``.

    Note:
        ``standardize=True`` se rechaza: la estandarización empírica depende de
        la muestra y los parámetros dejarían de ser exactos.
    """

    name = "mixture_exact"
    supported_dims = frozenset({2})

    def __init__(
        self,
        dim: int = 2,
        *,
        weights: Sequence[float] | np.ndarray,
        means: Sequence[Sequence[float]] | np.ndarray,
        covariances: Sequence[Sequence[Sequence[float]]] | np.ndarray,
        standardize: bool = False,
        seed: int | None = None,
    ) -> None:
        """Construye la mixtura validando sus parámetros de entrada.

        Args:
            dim: Dimensión del espacio; solo se admite ``2``.
            weights: Pesos ``(K,)``, no negativos y que suman uno.
            means: Medias ``(K, 2)``, una por componente.
            covariances: Covarianzas ``(K, 2, 2)``, cada una simétrica
                definida positiva.
            standardize: Debe ser ``False``.
            seed: Semilla del muestreo.

        Raises:
            ValueError: Si ``dim != 2``, ``standardize=True``, las shapes no
                coinciden en ``K``, hay pesos negativos o que no suman uno, o
                alguna covarianza no es simétrica definida positiva.
        """
        self._validate_dim(dim)
        if standardize:
            raise ValueError(
                "ExactGaussianMixture no admite standardize=True: la "
                "estandarización empírica estima media y desvío a partir de la "
                "muestra sorteada, así que los parámetros dependerían de n y de "
                "la semilla y dejarían de ser exactos. Usá standardize=False y, "
                "si querés datos centrados, declaralo en means/covariances."
            )
        super().__init__(dim, standardize=False, noise=0.0, seed=seed)

        # Copias: si el llamador muta sus buffers, no se rompen los invariantes.
        w = np.array(weights, dtype=np.float64)
        if w.ndim != 1 or w.size < 1:
            raise ValueError(f"weights debe ser un vector (K,) con K >= 1; recibí shape {w.shape}")
        k = int(w.shape[0])

        mu = np.array(means, dtype=np.float64)
        if mu.shape != (k, 2):
            raise ValueError(
                f"means debe tener shape (K, 2) con los K={k} componentes que "
                f"declara weights; recibí shape {mu.shape}"
            )

        cov = np.array(covariances, dtype=np.float64)
        if cov.shape != (k, 2, 2):
            raise ValueError(
                f"covariances debe tener shape (K, 2, 2) con los K={k} "
                f"componentes que declara weights; recibí shape {cov.shape}"
            )

        if np.any(w < 0.0):
            bad = np.flatnonzero(w < 0.0).tolist()
            raise ValueError(
                f"weights debe tener pesos no negativos; son negativos los "
                f"índices {bad}: {w[w < 0.0].tolist()}"
            )
        total = float(w.sum())
        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
            raise ValueError(f"weights debe sumar 1.0; recibí suma={total!r}")

        # Si Cholesky factoriza, es SPD; el factor se reusa al muestrear.
        chols = np.empty_like(cov)
        for i in range(k):
            c = cov[i]
            if not np.allclose(c, c.T, rtol=1e-10, atol=1e-12):
                raise ValueError(f"covariances[{i}] debe ser simétrica; recibí {c.tolist()}")
            try:
                chols[i] = np.linalg.cholesky(c)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"covariances[{i}] debe ser definida positiva (falló su "
                    f"factorización de Cholesky); recibí {c.tolist()}"
                ) from exc

        self._weights = w
        self._means = mu
        self._covariances = cov
        self._chols = chols

    # ------------------------------------------------- parámetros verdaderos

    @property
    def weights_(self) -> np.ndarray:
        """Pesos verdaderos ``(K,)`` float64 (copia)."""
        return self._weights.copy()

    @property
    def means_(self) -> np.ndarray:
        """Medias verdaderas ``(K, 2)`` float64 (copia)."""
        return self._means.copy()

    @property
    def covariances_(self) -> np.ndarray:
        """Covarianzas verdaderas ``(K, 2, 2)`` float64 (copia)."""
        return self._covariances.copy()

    # ------------------------------------------------ geometrías predefinidas

    @classmethod
    def two_modes(
        cls,
        *,
        separation: float,
        weights: tuple[float, float] = (0.5, 0.5),
        anisotropy: float = 1.0,
        scale: float = 0.3,
        angle: float = 0.0,
        seed: int | None = None,
    ) -> ExactGaussianMixture:
        """Mixtura de dos modos con separación, pesos y anisotropía por nombre.

        Medias simétricas respecto del origen: ``means = ∓½·separation·u`` con
        ``u = (cos angle, sin angle)``. Ambas componentes comparten covarianza,
        estirada a lo largo de ``u`` (los modos se alargan uno hacia el otro).

        Args:
            separation: Distancia entre las medias (>= 0); ``0`` superpone los
                modos.
            weights: Pesos ``(w0, w1)`` no negativos que suman uno.
            anisotropy: Razón mayor/menor de autovalores (>= 1); ver
                :func:`_anisotropic_covariance`.
            scale: Escala global de cada componente (> 0); con
                ``anisotropy=1`` la covarianza es ``scale**2 · I``.
            angle: Ángulo en radianes de la dirección de separación.
            seed: Semilla del muestreo.

        Raises:
            ValueError: Si ``separation < 0``, ``weights`` no tiene dos pesos,
                ``scale <= 0``, ``anisotropy < 1``, o los pesos son inválidos.
        """
        if not float(separation) >= 0.0:
            raise ValueError(
                f"separation debe ser >= 0 (es la distancia entre las dos "
                f"medias); recibí {separation!r}"
            )
        w = np.array(weights, dtype=np.float64)
        if w.shape != (2,):
            raise ValueError(
                f"weights debe tener exactamente 2 pesos, uno por modo; recibí shape {w.shape}"
            )
        _validate_scale_and_anisotropy(scale, anisotropy)

        direction = np.array([np.cos(float(angle)), np.sin(float(angle))], dtype=np.float64)
        half = 0.5 * float(separation)
        means = np.stack([-half * direction, half * direction])
        cov = _anisotropic_covariance(scale, anisotropy, float(angle))
        covariances = np.stack([cov, cov])
        return cls(2, weights=w, means=means, covariances=covariances, seed=seed)

    @classmethod
    def ring(
        cls,
        *,
        n_components: int = 8,
        radius: float = 5.0,
        scale: float = 0.3,
        weights: Sequence[float] | None = None,
        anisotropy: float = 1.0,
        seed: int | None = None,
    ) -> ExactGaussianMixture:
        """Mixtura en anillo (el clásico "8 gaussianas") con pesos y anisotropía.

        Medias en ``radius · (cos 2πk/K, sin 2πk/K)``, misma convención angular
        que :class:`GaussianMixture` en 2D. Cada componente se estira en su
        dirección radial, tomada del ángulo (definida aun con ``radius = 0``).

        Args:
            n_components: Cantidad de componentes ``K`` (>= 1).
            radius: Radio del anillo (>= 0).
            scale: Escala global de cada componente (> 0); con
                ``anisotropy=1`` la covarianza es ``scale**2 · I``.
            weights: Pesos ``(K,)`` que suman uno; ``None`` usa ``1/K``.
            anisotropy: Razón mayor/menor de autovalores (>= 1); ver
                :func:`_anisotropic_covariance`.
            seed: Semilla del muestreo.

        Raises:
            ValueError: Si ``n_components < 1``, ``radius < 0``, ``scale <= 0``,
                ``anisotropy < 1``, o ``weights`` es inválido.
        """
        k = int(n_components)
        if k < 1:
            raise ValueError(f"n_components debe ser >= 1; recibí {n_components!r}")
        if not float(radius) >= 0.0:
            raise ValueError(f"radius debe ser >= 0 (es el radio del anillo); recibí {radius!r}")
        _validate_scale_and_anisotropy(scale, anisotropy)

        if weights is None:
            w = np.full(k, 1.0 / k, dtype=np.float64)
        else:
            w = np.array(weights, dtype=np.float64)
            if w.shape != (k,):
                raise ValueError(
                    f"weights debe tener un peso por componente, es decir shape "
                    f"({k},) para n_components={k}; recibí shape {w.shape}"
                )

        angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
        means = float(radius) * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        covariances = np.stack([_anisotropic_covariance(scale, anisotropy, a) for a in angles])
        return cls(2, weights=w, means=means, covariances=covariances, seed=seed)

    # ------------------------------------------------------------- muestreo

    def _sample_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Muestrea ``n`` puntos y fija la etiqueta de componente en ``color_``.

        Las cantidades por componente salen de :func:`_largest_remainder_counts`
        (composición exacta) y los puntos quedan agrupados por componente, sin
        permutar.
        """
        counts = _largest_remainder_counts(self._weights, n)
        points, labels = [], []
        for i, count in enumerate(counts.tolist()):
            if count == 0:
                continue
            z = rng.standard_normal(size=(count, 2))
            points.append(self._means[i] + z @ self._chols[i].T)
            labels.append(np.full(count, i, dtype=np.int64))
        self.color_ = np.concatenate(labels, axis=0)
        return np.concatenate(points, axis=0)


class TwoMoons(PointDistribution):
    """Dos medias lunas entrelazadas (2D). Vía ``sklearn.make_moons``."""

    name = "two_moons"
    supported_dims = frozenset({2})

    def __init__(self, dim=2, *, noise=0.05, standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=noise, seed=seed)

    def _sample_raw(self, n, rng):
        x, y = make_moons(n_samples=n, noise=self.noise, random_state=_seed_from(rng))
        self.color_ = y
        return x


class Spiral(PointDistribution):
    """Una o más espirales entrelazadas (2D). Implementada con numpy."""

    name = "spiral"
    supported_dims = frozenset({2})

    def __init__(self, dim=2, *, noise=0.02, n_arms=2, turns=1.5, standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=noise, seed=seed)
        if n_arms < 1:
            raise ValueError(f"n_arms debe ser >= 1; recibí {n_arms}")
        self.n_arms = int(n_arms)
        self.turns = float(turns)

    def _sample_raw(self, n, rng):
        max_t = self.turns * 2.0 * np.pi
        arms, labels = [], []
        for i, k in enumerate(_split_counts(n, self.n_arms)):
            # sqrt(U) => densidad ~uniforme en área a lo largo del radio.
            t = np.sqrt(rng.uniform(0.0, 1.0, size=k)) * max_t
            phase = i * (2.0 * np.pi / self.n_arms)
            xy = np.stack([t * np.cos(t + phase), t * np.sin(t + phase)], axis=1)
            xy = xy / max_t  # normalizar a ~[-1, 1]
            xy = xy + rng.normal(0.0, self.noise, size=(k, 2))
            arms.append(xy)
            labels.append(np.full(k, i))
        out = np.concatenate(arms, axis=0)
        lab = np.concatenate(labels, axis=0)
        perm = rng.permutation(len(out))
        self.color_ = lab[perm]
        return out[perm]


class SwissRoll(PointDistribution):
    """Rollo suizo: variedad 2D embebida en 3D. Vía ``sklearn.make_swiss_roll``."""

    name = "swiss_roll"
    supported_dims = frozenset({3})

    def __init__(self, dim=3, *, noise=0.5, standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=noise, seed=seed)

    def _sample_raw(self, n, rng):
        x, t = make_swiss_roll(n_samples=n, noise=self.noise, random_state=_seed_from(rng))
        self.color_ = t
        return x
