"""Oráculo exacto de una mixtura gaussiana 2D ruideada por una SDE escalar-gaussiana.

Si ``p_0 = Σ_k w_k N(μ_k, Σ_k)`` y el kernel es ``N(α_t x_0, σ_t² I)``, la marginal sigue
siendo una mixtura:

``p_t = Σ_k w_k N(α_t μ_k, Σ_k(t))``  con  ``Σ_k(t) = α_t² Σ_k + σ_t² I``,

de donde salen en forma cerrada densidad, log-densidad y score. ``α_t`` y ``σ_t`` se leen
de ``marginal_prob`` evaluado sobre un vector de unos, sin ramificar por variante. sub-VP
publica su desvío sin raíz (``std = 1 - e^{-∫β}``) y se usa tal cual; VE tiene ``α_t = 1``.

El sesgo de inicialización ``KL(p_T ‖ q)`` no tiene forma cerrada (la entropía de una
mixtura gaussiana no la tiene), así que se reportan una cota superior por convexidad,
``KL(Σ_k w_k N_k ‖ q) ≤ Σ_k w_k KL(N_k ‖ q)``, y un valor de referencia por cuadratura 2D.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import torch

from ..data_generation import ExactGaussianMixture
from ..sde import ForwardSDE
from .quadrature import QuadratureGrid, auto_grid, integrate

#: Término constante de la log-densidad gaussiana en dos dimensiones: ``-(d/2) log(2π)``.
_LOG_NORMALIZACION_2D: float = -math.log(2.0 * math.pi)

#: Puntos de sondeo del chequeo de familia: coordenadas no nulas (para despejar ``α_t``)
#: y distintas entre sí (un vector de unos no detectaría un kernel que mezcle coordenadas).
_PROBE_X0: tuple[tuple[float, float], ...] = ((1.0, 2.0), (3.0, -1.5))

#: Fracciones del horizonte en las que se sondea el contrato marginal.
_PROBE_FRACTIONS: tuple[float, ...] = (0.25, 0.75)

#: Tolerancia relativa con la que se exige ``mean == α_t · x_0`` en el sondeo.
_PROPORTIONALITY_TOL: float = 1e-10

#: Dimensión del espacio (explícita en la KL gaussiana: términos ``-d`` y ``d·log σ_q²``).
_DIM: int = 2

#: Desvíos del estimador que cubre la tolerancia de
#: :meth:`MixtureOracle.estimate_prior_variance` (probabilidad de excederla ~``1e-4``).
_TOLERANCE_SIGMAS: float = 4.0


@dataclass(frozen=True, kw_only=True)
class BiasReport:
    """Sesgo de inicialización: cuánto se aparta ``p_T`` del ruido de partida.

    Attributes:
        bound: Cota superior de forma cerrada por convexidad de la KL (un techo, no una
            estimación).
        value: Valor de referencia de la divergencia por cuadratura. ``<= bound`` salvo
            cuando ambos son cero a nivel de redondeo.
        tolerance: Cota del error de ``value`` (ver :func:`_quadrature_tolerance`).
        mass: Masa integrada sobre la misma malla que ``value``; debe dar uno.
        prior_variance: Varianza de la distribución de partida declarada por el caller.
        horizon: Instante en el que se evaluó el sesgo.
    """

    bound: float
    value: float
    tolerance: float
    mass: float
    prior_variance: float
    horizon: float

    def __post_init__(self) -> None:
        """Valida que todas las cantidades sean finitas y no negativas (``prior_variance > 0``)."""
        if not math.isfinite(self.bound) or self.bound < 0.0:
            raise ValueError(
                "bound debe ser finito y no negativo: es una divergencia acotada por "
                f"convexidad y cada término es una KL; recibí {self.bound!r}."
            )
        if not math.isfinite(self.prior_variance) or self.prior_variance <= 0.0:
            raise ValueError(
                "prior_variance debe ser finita y estrictamente positiva; recibí "
                f"{self.prior_variance!r}."
            )
        if not math.isfinite(self.horizon) or self.horizon < 0.0:
            raise ValueError(
                "horizon debe ser finito y no negativo: el proceso corre en [0, T]; recibí "
                f"{self.horizon!r}."
            )
        for nombre, valor in (
            ("value", self.value),
            ("tolerance", self.tolerance),
            ("mass", self.mass),
        ):
            if not math.isfinite(valor) or valor < 0.0:
                raise ValueError(
                    f"{nombre} debe ser finito y no negativo: sale de una cuadratura sobre "
                    f"una densidad no negativa; recibí {valor!r}."
                )


class MixtureOracle:
    """Densidad, score y sesgo exactos de una mixtura 2D ruideada por una SDE escalar-gaussiana.

    Determinístico y sin estado mutable; la admisibilidad se valida en construcción.

    Note:
        Solo describe mixturas exactas: con estandarización empírica los parámetros
        efectivos dependerían de la muestra.
    """

    def __init__(self, mixture: ExactGaussianMixture, sde: ForwardSDE) -> None:
        """Construye el oráculo validando mixtura y SDE.

        Args:
            mixture: Mixtura de parámetros exactos.
            sde: Proceso forward escalar-gaussiano; solo se usan ``marginal_prob`` y ``T``.

        Raises:
            ValueError: Si ``mixture`` usa ``standardize=True`` o no es una
                :class:`ExactGaussianMixture`.
            NotImplementedError: Si ``sde`` no es de la familia escalar-gaussiana (verificado
                sobre lo que devuelve ``marginal_prob``).
        """
        if getattr(mixture, "standardize", False):
            raise ValueError(
                "mixture no puede usar la estandarización empírica (standardize=True): "
                "esa transformación se estima a partir de la muestra sorteada, así que los "
                "parámetros efectivos dependen del sorteo (de n y de la semilla) y el "
                "resultado no sería exacto. Construí una ExactGaussianMixture con los "
                f"parámetros que querés y standardize=False; recibí "
                f"{type(mixture).__name__} con standardize=True."
            )
        if not isinstance(mixture, ExactGaussianMixture):
            raise ValueError(
                "mixture debe ser una ExactGaussianMixture, la única fuente de datos que "
                "publica sus pesos, medias y covarianzas verdaderos en forma cerrada; "
                f"recibí {type(mixture).__name__}, cuyos parámetros no son consultables, "
                "así que ninguna cantidad derivada de ellos sería exacta."
            )

        self._mixture = mixture
        self._sde = sde

        # Los accesores ya devuelven copias: estos tensores no aliasan nada del caller.
        self._weights = torch.as_tensor(mixture.weights_, dtype=torch.float64)
        self._means = torch.as_tensor(mixture.means_, dtype=torch.float64)
        self._covariances = torch.as_tensor(mixture.covariances_, dtype=torch.float64)

        self._validate_scalar_gaussian_family()

    # ------------------------------------------------- parámetros del kernel en t

    def marginal_params(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Factor de contracción ``α_t`` y desvío del kernel ``σ_t``.

        Evalúa ``marginal_prob`` sobre un vector de unos en doble precisión: para la familia
        escalar-gaussiana ``mean == α_t`` y ``std == σ_t``.

        Args:
            t: Tiempos ``(B,)`` o ``(B, 1)``, finitos y no negativos.

        Returns:
            ``(alpha, sigma)``, cada uno ``(B, 1)`` float64 en el device de ``t``. ``sigma`` es
            el desvío tal como lo publica la SDE (para sub-VP, sin raíz).

        Raises:
            ValueError: Si ``t`` tiene forma inválida o valores no finitos o negativos.
        """
        tt = _normalize_time(t)
        batch = tt.shape[0]
        ones = torch.ones(batch, 2, dtype=torch.float64, device=tt.device)
        mean, std = self._sde.marginal_prob(ones, tt)
        return _as_column(mean, batch), _as_column(std, batch)

    def component_covariances(self, t: torch.Tensor) -> torch.Tensor:
        """Covarianzas ``Σ_k(t) = α_t² Σ_k + σ_t² I``, de forma ``(B, K, 2, 2)`` float64."""
        alpha, sigma = self.marginal_params(t)
        return _perturbed_covariances(self._covariances.to(device=alpha.device), alpha, sigma)

    # ------------------------------------------------- densidad y log-densidad

    def log_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Log-densidad exacta ``log p_t(x)``.

        Se calcula con ``logsumexp`` sobre ``log w_k + log N(x; α_t μ_k, Σ_k(t))``, que no
        desborda a ``-inf`` lejos de los modos cuando ``σ_t`` es chico. Derivable (dos veces)
        respecto de ``x``; no respecto de ``t``. Cálculo interno en float64.

        Args:
            x: Puntos ``(B, 2)`` de punto flotante.
            t: Tiempos ``(B,)`` o ``(B, 1)``, uno por punto.

        Returns:
            Tensor ``(B,)`` en el dtype y device de ``x``.

        Raises:
            ValueError: Si ``x`` o ``t`` tienen forma o dtype inválidos, o los lotes no coinciden.
        """
        return self._log_prob_double(x, t).to(x.dtype)

    def prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Densidad exacta ``p_t(x)``, como ``exp`` de :meth:`log_prob`."""
        return torch.exp(self._log_prob_double(x, t)).to(x.dtype)

    # ------------------------------------------------------------------- score

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score exacto ``∇_x log p_t(x)``::

            ∇ log p_t(x) = -Σ_k r_k(x, t) · Σ_k(t)^{-1} (x - α_t μ_k)

        donde ``r_k`` son las responsabilidades posteriores, calculadas con ``softmax`` sobre
        los mismos términos ``log w_k + log N_k`` que :meth:`log_prob` (evita ``0/0`` cuando
        ``σ_t`` es chico y garantiza consistencia con su gradiente).

        Note:
            A diferencia de ``ForwardSDE.score_target``, ``σ_t`` no se recorta por un piso: es
            el valor exacto, también para ``t → 0``. Sigue siendo finito porque ``Σ_k(t)`` tiene
            piso ``α_t² λ_min(Σ_k)``.

        Args:
            x: Puntos ``(B, 2)`` de punto flotante.
            t: Tiempos ``(B,)`` o ``(B, 1)``, uno por punto.

        Returns:
            Tensor ``(B, 2)`` en el dtype y device de ``x``; cálculo interno en float64.

        Raises:
            ValueError: Si ``x`` o ``t`` tienen forma o dtype inválidos, o los lotes no coinciden.
        """
        log_pesados, empuje = self._component_terms(x, t)
        # Responsabilidades posteriores (B, K).
        responsabilidades = torch.softmax(log_pesados, dim=1)
        score = -(responsabilidades.unsqueeze(-1) * empuje).sum(dim=1)
        return score.to(x.dtype)

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Alias de :meth:`score`; cumple el contrato ``(x, t) -> score`` de los samplers."""
        return self.score(x, t)

    def total_mass(self, t: float, *, grid: QuadratureGrid | None = None) -> float:
        """Masa ``∫ p_t`` por cuadratura: autochequeo del oráculo (debe dar uno).

        Args:
            t: Tiempo escalar, finito y no negativo.
            grid: Malla explícita; por defecto ``auto_grid`` sobre los parámetros en ``t``.

        Raises:
            ValueError: Si ``t`` no es un número real finito y no negativo.
        """
        instante = _normalize_scalar_time(t)
        medias, covarianzas = self._propagated_components(instante)
        malla = grid if grid is not None else auto_grid(means=medias, covariances=covarianzas)

        def densidad(puntos: torch.Tensor) -> torch.Tensor:
            """Densidad evaluada en los nodos de un bloque de la malla."""
            tiempos = torch.full(
                (puntos.shape[0],),
                instante,
                dtype=puntos.dtype,
                device=puntos.device,
            )
            return self.prob(puntos, tiempos)

        return integrate(densidad, malla)

    # ------------------------------------------------- sesgo de inicialización

    def initialization_bias(
        self,
        prior_variance: float,
        *,
        t: float | None = None,
        grid: QuadratureGrid | None = None,
    ) -> BiasReport:
        """Sesgo de inicialización ``KL(p_t ‖ N(0, σ_q² I))``: cota cerrada y valor por cuadratura.

        La cota es ``Σ_k w_k KL(N_k ‖ q)`` con cada término en forma cerrada::

            KL( N(m, S) ‖ N(0, σ_q² I) ) = ½[ tr(S)/σ_q² + mᵀm/σ_q² − 2 − log det S
                                              + 2 log σ_q² ]

        sobre ``m = α_t μ_k`` y ``S = Σ_k(t)``; con ``K = 1`` es exacta. El valor es::

            value = ∫ p_t(x) · [ log p_t(x) − log N(x; 0, σ_q² I) ] dx

        sobre una malla ampliada a ``prior_std = sqrt(prior_variance)``, y se recorta a cero
        por abajo para absorber ruido de redondeo.

        Args:
            prior_variance: Varianza de la distribución de partida (obligatoria; no se puede
                leer de la SDE). ``1.0`` para VP y sub-VP, ``σ_max²`` para VE (``25.0`` por
                defecto). Ver :meth:`estimate_prior_variance`.
            t: Instante a evaluar; por defecto el horizonte ``T`` de la SDE.
            grid: Malla explícita para la cuadratura; no afecta a la cota.

        Raises:
            ValueError: Si ``prior_variance`` no es real, finita y positiva, si ``t`` es
                inválido, o si la masa integrada se aparta de uno más que la tolerancia (la
                malla no alcanza).
        """
        varianza = _normalize_prior_variance(prior_variance)
        horizonte = float(self._sde.T) if t is None else _normalize_scalar_time(t)

        medias, covarianzas = self._propagated_components(horizonte)
        por_componente = _kl_against_isotropic(medias, covarianzas, varianza)
        cota = float((self._weights * por_componente).sum())

        malla = (
            grid
            if grid is not None
            else auto_grid(
                means=medias,
                covariances=covarianzas,
                prior_std=math.sqrt(varianza),
            )
        )
        # La masa va primero: si la malla no alcanza, se falla sin integrar el valor.
        masa = self.total_mass(horizonte, grid=malla)
        desviacion = abs(masa - 1.0)
        tolerancia = _quadrature_tolerance(malla, masa)
        if desviacion > tolerancia:
            raise ValueError(
                f"la masa integrada dio {masa!r} y se aparta de uno en {desviacion!r}, más que "
                f"la tolerancia {tolerancia!r} que una malla de {malla.n_points} nodos por eje "
                f"sobre [-{malla.half_width}, {malla.half_width}] admite: la malla no alcanza "
                "para esta densidad. El valor de referencia calculado sobre ella no sería "
                "confiable, así que no se devuelve. Ampliá el dominio o refiná el paso (o dejá "
                "que se dimensione automáticamente omitiendo grid)."
            )

        def divergencia_puntual(puntos: torch.Tensor) -> torch.Tensor:
            """Integrando ``p_t (log p_t − log q)`` en los nodos de un bloque de la malla."""
            tiempos = torch.full(
                (puntos.shape[0],),
                horizonte,
                dtype=puntos.dtype,
                device=puntos.device,
            )
            log_p = self._log_prob_double(puntos, tiempos)
            log_q = _log_isotropic_gaussian(puntos.to(torch.float64), varianza)
            densidad = torch.exp(log_p)
            # Límite p log p -> 0; evita 0 · (-inf) = nan donde la densidad desborda a cero.
            return torch.where(
                densidad > 0.0,
                densidad * (log_p - log_q),
                torch.zeros_like(densidad),
            )

        valor = max(integrate(divergencia_puntual, malla), 0.0)

        return BiasReport(
            bound=cota,
            value=valor,
            tolerance=tolerancia,
            mass=masa,
            prior_variance=varianza,
            horizon=horizonte,
        )

    def estimate_prior_variance(self, *, n: int = 1_000_000, seed: int = 0) -> tuple[float, float]:
        """Estima la varianza de la distribución de partida muestreándola (opt-in).

        Estima ``E[x²]`` por coordenada (los priors están centrados). La tolerancia relativa es
        ``_TOLERANCE_SIGMAS · sqrt(2/n)``, ya que ``Var(x²) = 2v²`` para ``x ~ N(0, v)``. Usa
        un generador propio, sin tocar el RNG global.

        Args:
            n: Cantidad de puntos a muestrear.
            seed: Semilla del generador.

        Returns:
            ``(varianza, tolerancia_relativa)``.

        Raises:
            ValueError: Si ``n`` no es un entero >= 2 o ``seed`` no es entero.
        """
        if not isinstance(n, numbers.Integral) or int(n) < 2:
            raise ValueError(
                "n debe ser un entero de al menos 2 (es una cantidad de puntos a "
                f"muestrear); recibí {n!r}."
            )
        if not isinstance(seed, numbers.Integral):
            raise ValueError(
                "seed debe ser un entero para sembrar el generador; recibí un objeto de "
                f"tipo {type(seed).__name__}."
            )

        cantidad = int(n)
        generador = torch.Generator().manual_seed(int(seed))
        muestra = self._sde.prior_sampling(
            (cantidad, _DIM), generator=generador, dtype=torch.float64
        )
        varianza = float((muestra.to(torch.float64) ** 2).mean())
        tolerancia = _TOLERANCE_SIGMAS * math.sqrt(2.0 / cantidad)
        return varianza, tolerancia

    # ------------------------------------------------------------------ internos

    def _propagated_components(self, instante: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Medias ``α_t μ_k`` ``(K, 2)`` y covarianzas ``Σ_k(t)`` ``(K, 2, 2)`` en un instante escalar."""
        tt = torch.tensor([[instante]], dtype=torch.float64)
        alpha, sigma = self.marginal_params(tt)
        covarianzas = _perturbed_covariances(self._covariances, alpha, sigma)[0]
        medias = (alpha.reshape(-1, 1, 1) * self._means)[0]
        return medias, covarianzas

    def _log_prob_double(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Log-densidad ``(B,)`` en float64, conectada al grafo de ``x``."""
        log_pesados, _ = self._component_terms(x, t)
        return torch.logsumexp(log_pesados, dim=1)

    def _component_terms(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Términos por componente de los que salen densidad y score.

        Returns:
            ``(log_pesados, empuje)``: ``log w_k + log N_k`` de forma ``(B, K)`` y
            ``Σ_k(t)^{-1} (x - α_t μ_k)`` de forma ``(B, K, 2)``, en float64 y en el device de
            ``x``.

        Raises:
            ValueError: Si ``x`` o ``t`` no cumplen su contrato, o si los lotes no coinciden.
        """
        xx = _normalize_state(x)
        tt = _normalize_time(t).to(device=xx.device)
        if tt.shape[0] != xx.shape[0]:
            raise ValueError(
                "t debe traer un tiempo por punto de x (un valor por fila); recibí "
                f"{tt.shape[0]} tiempos para {xx.shape[0]} puntos."
            )

        alpha, sigma = self.marginal_params(tt)  # (B, 1) cada uno
        base = self._covariances.to(device=xx.device)  # (K, 2, 2)
        covarianzas = _perturbed_covariances(base, alpha, sigma)  # (B, K, 2, 2)
        precisiones, determinantes = _inverse_and_det_2x2(covarianzas)

        # Medias contraídas por el kernel: alpha_t mu_k, con forma (B, K, 2).
        medias = alpha.reshape(-1, 1, 1) * self._means.to(device=xx.device)
        diferencia = xx.unsqueeze(1) - medias  # (B, K, 2)
        # Sigma_k(t)^{-1} (x - alpha_t mu_k), (B, K, 2): lo usan el score y la forma cuadrática.
        empuje = (precisiones @ diferencia.unsqueeze(-1)).squeeze(-1)
        # Forma cuadrática (x - mu_k)^T Sigma_k(t)^{-1} (x - mu_k), de forma (B, K).
        cuadratica = (diferencia * empuje).sum(dim=-1)

        log_componentes = _LOG_NORMALIZACION_2D - 0.5 * torch.log(determinantes) - 0.5 * cuadratica
        # Un peso nulo da -inf, que logsumexp absorbe sin nan; no se recorta a propósito.
        log_pesos = torch.log(self._weights.to(device=xx.device))  # (K,)
        return log_pesos + log_componentes, empuje

    def _validate_scalar_gaussian_family(self) -> None:
        """Verifica que el kernel sea ``N(α_t x_0, σ_t² I)``.

        Sondea ``marginal_prob`` y exige media proporcional a ``x_0`` con un escalar por
        muestra y desvío escalar por muestra.

        Raises:
            NotImplementedError: Si alguna de las dos propiedades no se cumple.
        """
        x0 = torch.tensor(_PROBE_X0, dtype=torch.float64)
        horizonte = float(self._sde.T)
        t = torch.tensor([f * horizonte for f in _PROBE_FRACTIONS], dtype=torch.float64).reshape(
            -1, 1
        )
        mean, std = self._sde.marginal_prob(x0, t)

        if not _collapses_to_scalar_per_sample(std, x0.shape[0]):
            raise NotImplementedError(
                "sde no pertenece a la familia escalar-gaussiana: su contrato marginal "
                f"devolvió un desvío de forma {tuple(std.shape)}, que no colapsa a un "
                f"escalar por muestra para un lote de {x0.shape[0]}. Este oráculo solo "
                "describe kernels N(alpha_t x_0, sigma_t^2 I); una SDE con desvío por "
                "coordenada o con estado aumentado necesita su propia forma cerrada."
            )
        if tuple(mean.shape) != tuple(x0.shape):
            raise NotImplementedError(
                "sde no pertenece a la familia escalar-gaussiana: su contrato marginal "
                f"devolvió una media de forma {tuple(mean.shape)} para un dato de forma "
                f"{tuple(x0.shape)}."
            )
        alpha = mean[:, :1] / x0[:, :1]
        if not torch.allclose(mean, alpha * x0, rtol=_PROPORTIONALITY_TOL, atol=1e-12):
            raise NotImplementedError(
                "sde no pertenece a la familia escalar-gaussiana: su contrato marginal no "
                "devolvió una media proporcional a x_0 (se espera mean == alpha_t * x_0 "
                "con alpha_t escalar por muestra, que es lo que hace recuperable alpha_t "
                "sondeando con un vector de unos)."
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return f"{type(self).__name__}(K={self._weights.shape[0]}, sde={type(self._sde).__name__})"


# ---------------------------------------------------------------- helpers numéricos


def _normalize_time(t: torch.Tensor) -> torch.Tensor:
    """Valida ``t`` ``(B,)`` o ``(B, 1)`` y lo devuelve ``(B, 1)`` float64, desconectado del grafo.

    Raises:
        ValueError: Si ``t`` no es tensor, tiene forma inválida o valores no finitos o
            negativos.
    """
    if not isinstance(t, torch.Tensor):
        raise ValueError(
            "t debe ser un tensor de torch de forma (B,) o (B, 1) con un tiempo por "
            f"punto; recibí un objeto de tipo {type(t).__name__}."
        )
    forma_valida = t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1)
    if not forma_valida or t.shape[0] < 1:
        raise ValueError(
            "t debe tener forma (B,) o (B, 1) con B >= 1 (un tiempo por punto); recibí "
            f"forma {tuple(t.shape)}."
        )
    tt = t.detach().to(torch.float64).reshape(t.shape[0], 1)
    if not bool(torch.isfinite(tt).all()) or bool((tt < 0.0).any()):
        raise ValueError(
            "t debe ser finito y no negativo: el proceso corre en [0, T] y fuera de ese "
            "rango el schedule de la SDE no está definido."
        )
    return tt


def _normalize_scalar_time(t: float) -> float:
    """Valida un tiempo escalar y lo devuelve como ``float``.

    Se valida por tipo (no con ``float()``) para rechazar cadenas como ``"0.3"``.

    Raises:
        ValueError: Si ``t`` no es un número real, o no es finito y no negativo.
    """
    if not isinstance(t, numbers.Real):
        raise ValueError(
            "t debe ser un número real que exprese un instante del proceso; recibí un "
            f"objeto de tipo {type(t).__name__}."
        )
    instante = float(t)
    if not math.isfinite(instante) or instante < 0.0:
        raise ValueError(
            "t debe ser finito y no negativo: el proceso corre en [0, T] y fuera de ese "
            f"rango el schedule de la SDE no está definido; recibí {instante!r}."
        )
    return instante


def _normalize_prior_variance(prior_variance: float) -> float:
    """Valida la varianza de partida (número real, finita y positiva) y la devuelve como ``float``.

    Raises:
        ValueError: Si no es un número real, o no es finita y estrictamente positiva.
    """
    if not isinstance(prior_variance, numbers.Real):
        raise ValueError(
            "prior_variance debe ser un número real con la varianza de la distribución de "
            "la que arranca el sampler (1.0 para VP y sub-VP, sigma_max**2 para VE); recibí "
            f"un objeto de tipo {type(prior_variance).__name__}."
        )
    valor = float(prior_variance)
    if not math.isfinite(valor) or valor <= 0.0:
        raise ValueError(
            "prior_variance debe ser finita y estrictamente positiva: una gaussiana de "
            f"varianza nula o negativa no describe ninguna distribución de partida; recibí "
            f"{valor!r}."
        )
    return valor


def _kl_against_isotropic(
    means: torch.Tensor, covariances: torch.Tensor, prior_variance: float
) -> torch.Tensor:
    """``KL(N(m_k, S_k) ‖ N(0, σ_q² I))`` de cada componente, en forma cerrada::

        ½[ tr(S)/σ_q² + mᵀm/σ_q² − d − log det S + d·log σ_q² ]

    Cada término se recorta a cero por abajo (ruido de redondeo cuando la componente
    coincide con el prior).

    Args:
        means: Medias ya contraídas ``(K, 2)``.
        covariances: Covarianzas ya propagadas ``(K, 2, 2)``.
        prior_variance: Varianza de la distribución de partida.

    Returns:
        Tensor ``(K,)`` no negativo.

    Raises:
        ValueError: Si alguna covarianza no tiene determinante finito y positivo.
    """
    _, determinantes = _inverse_and_det_2x2(covariances)
    trazas = covariances.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    normas2 = (means * means).sum(dim=-1)
    divergencias = 0.5 * (
        (trazas + normas2) / prior_variance
        - float(_DIM)
        - torch.log(determinantes)
        + _DIM * math.log(prior_variance)
    )
    return divergencias.clamp_min(0.0)


def _log_isotropic_gaussian(x: torch.Tensor, variance: float) -> torch.Tensor:
    """``log N(x; 0, σ_q² I)`` en 2D para puntos ``(N, 2)``; devuelve ``(N,)``."""
    return (
        _LOG_NORMALIZACION_2D
        - 0.5 * _DIM * math.log(variance)
        - (x * x).sum(dim=-1) / (2.0 * variance)
    )


def _quadrature_tolerance(grid: QuadratureGrid, mass: float) -> float:
    """Cota adimensional del error de la cuadratura::

        tolerance = h · ( h + |mass − 1| ),   h = spacing / (2 half_width)

    ``h²`` acota el error de discretización (conservador: sobre gaussianas la convergencia
    es espectral). ``h · |mass − 1|`` cubre lo que el dominio no abarca; el factor ``h``
    hace que exigir ``|mass − 1| <= tolerance`` sea un chequeo no trivial.
    """
    paso_relativo = grid.spacing / (2.0 * grid.half_width)
    return paso_relativo * (paso_relativo + abs(mass - 1.0))


def _normalize_state(x: torch.Tensor) -> torch.Tensor:
    """Valida ``x`` ``(B, 2)`` de punto flotante y lo pasa a float64 sin cortar el grafo.

    Raises:
        ValueError: Si ``x`` no es tensor, no tiene forma ``(B, 2)`` o no es de punto
            flotante.
    """
    if not isinstance(x, torch.Tensor):
        raise ValueError(
            "x debe ser un tensor de torch de forma (B, 2) con un punto del plano por "
            f"fila; recibí un objeto de tipo {type(x).__name__}."
        )
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 1:
        raise ValueError(
            "x debe tener forma (B, 2) con B >= 1 (el laboratorio analítico es 2D); "
            f"recibí forma {tuple(x.shape)}."
        )
    if not x.dtype.is_floating_point:
        raise ValueError(
            "x debe ser un tensor de punto flotante: un estado entero no describe un punto "
            f"del plano; recibí dtype {x.dtype}."
        )
    return x.to(torch.float64)


def _perturbed_covariances(
    base: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """``Σ_k(t) = α_t² Σ_k + σ_t² I``: ``(K, 2, 2)`` y ``(B, 1)`` → ``(B, K, 2, 2)``."""
    identidad = torch.eye(2, dtype=base.dtype, device=base.device)
    alpha2 = (alpha**2).reshape(-1, 1, 1, 1)  # (B, 1, 1, 1)
    sigma2 = (sigma**2).reshape(-1, 1, 1, 1)
    return alpha2 * base + sigma2 * identidad


def _as_column(value: torch.Tensor, batch: int) -> torch.Tensor:
    """Lleva un escalar por muestra (``(B, 1, …)``, ``(1, 1, …)`` o rango cero) a ``(batch, 1)``."""
    if value.ndim == 0:
        return value.reshape(1, 1).expand(batch, 1)
    columna = value.reshape(value.shape[0], -1)[:, :1]
    if columna.shape[0] != batch:
        columna = columna.expand(batch, 1)
    return columna


def _collapses_to_scalar_per_sample(value: torch.Tensor, batch: int) -> bool:
    """Indica si ``value`` es un escalar por muestra (lote ``1`` o ``batch``, resto de dims ``1``)."""
    if value.ndim == 0:
        return True
    if value.shape[0] not in (1, batch):
        return False
    return all(d == 1 for d in value.shape[1:])


def _inverse_and_det_2x2(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inversa y determinante de matrices ``(…, 2, 2)`` en forma cerrada (adjunta / ``ad - bc``).

    Raises:
        ValueError: Si la forma no termina en ``(2, 2)`` o algún determinante no es finito
            y positivo.
    """
    if cov.ndim < 2 or cov.shape[-2:] != (2, 2):
        raise ValueError(f"cov debe tener forma (..., 2, 2); recibida {tuple(cov.shape)}.")
    a = cov[..., 0, 0]
    b = cov[..., 0, 1]
    c = cov[..., 1, 0]
    d = cov[..., 1, 1]
    det = a * d - b * c
    if not bool(torch.isfinite(det).all()) or bool((det <= 0.0).any()):
        raise ValueError(
            "cov debe tener determinante finito y positivo para ser invertible como "
            f"covarianza; el mínimo encontrado fue {float(det.min())!r}."
        )
    fila0 = torch.stack((d, -b), dim=-1)
    fila1 = torch.stack((-c, a), dim=-1)
    adjunta = torch.stack((fila0, fila1), dim=-2)
    return adjunta / det[..., None, None], det
