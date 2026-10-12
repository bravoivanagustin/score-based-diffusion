"""Fuente de datos de imágenes sin labels (p. ej. fotos de gatos).

Convierte una carpeta de imágenes en batches de tensores ``(B, 3, H, W)`` en
``[-1, 1]``. ``torch``, ``torchvision`` y ``PIL`` se importan de forma diferida
para que ``import diffusion.data_generation`` siga dependiendo solo de numpy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # solo para type-checkers; no se ejecuta en runtime
    import torch

#: Extensiones reconocidas como imagen (se comparan en minúsculas).
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def _discover_image_paths(root: str | Path) -> list[Path]:
    """Lista ordenada (recursiva) de los archivos de imagen bajo ``root``.

    Raises:
        ValueError: Si ``root`` no existe o no contiene ninguna imagen.
    """
    root = Path(root)
    if not root.exists():
        raise ValueError(f"La carpeta raíz de imágenes no existe: {root!s}")
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        exts = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(f"No se encontraron imágenes en {root!s} (extensiones buscadas: {exts}).")
    return paths


def _build_transform(
    image_size: int,
    augment: bool,
    crop: bool = True,
) -> Callable[..., torch.Tensor]:
    """Arma la cadena PIL RGB → tensor ``(3, image_size, image_size)`` en ``[-1, 1]``.

    Orden: (flip horizontal opcional) → encuadre → ``ToTensor`` → ``Normalize``.
    No se usan volteos verticales ni rotaciones: no son muestras válidas.

    Args:
        image_size: Lado del cuadrado de salida en píxeles.
        augment: Si es ``True``, antepone ``RandomHorizontalFlip(p=0.5)``.
        crop: ``True`` preserva el aspect ratio (``Resize`` del lado corto +
            ``CenterCrop``); ``False`` deforma al cuadrado sin recortar.
    """
    from torchvision import transforms

    steps: list[Callable[..., object]] = []
    if augment:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    if crop:
        steps.append(transforms.Resize(image_size))
        steps.append(transforms.CenterCrop(image_size))
    else:
        steps.append(transforms.Resize((image_size, image_size)))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    return transforms.Compose(steps)


def _build_cat_images_class() -> type:
    """Construye la clase ``CatImages`` (importa torch recién al llamarse)."""
    import torch

    class CatImages(torch.utils.data.Dataset):
        """Dataset de imágenes sin labels: ``__getitem__`` devuelve solo el tensor.

        Args:
            root: Carpeta raíz con las imágenes (se recorre recursivamente).
            transform: Callable ``PIL.Image`` RGB → ``torch.Tensor``.

        Raises:
            ValueError: Si ``root`` no existe o no contiene ninguna imagen.
        """

        def __init__(
            self,
            root: str | Path,
            transform: Callable[..., torch.Tensor],
        ) -> None:
            self.root = Path(root)
            self.transform = transform
            self.paths: list[Path] = _discover_image_paths(root)

        def __len__(self) -> int:
            """Cantidad de imágenes descubiertas bajo ``root``."""
            return len(self.paths)

        def __getitem__(self, index: int) -> torch.Tensor:
            """Lee la imagen ``index``, la pasa a RGB y le aplica el transform.

            ``convert("RGB")`` garantiza 3 canales aunque la original sea gris,
            RGBA o CMYK.
            """
            from PIL import Image

            with Image.open(self.paths[index]) as img:
                image = img.convert("RGB")
            return self.transform(image)

    return CatImages


def _infinite(loader) -> Iterator[torch.Tensor]:
    """Recorre un ``DataLoader`` finito en bucle, indefinidamente.

    A diferencia de :func:`infinite_bare`, no desempaqueta nada: el loader ya
    yield-ea tensores ``(B, 3, H, W)``.
    """
    while True:
        yield from loader


def infinite_batches(
    root: str | Path,
    batch_size: int,
    *,
    image_size: int = 64,
    augment: bool = True,
    crop: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int | None = None,
    pin_memory: bool = False,
) -> Iterator[torch.Tensor]:
    """Fuente infinita de batches de imágenes para entrenar.

    Cada ``next()`` devuelve un tensor ``(batch_size, 3, image_size, image_size)``
    float32 en ``[-1, 1]``. Usa ``drop_last=True``, así que todos los batches
    tienen tamaño exacto.

    Args:
        root: Carpeta raíz con las imágenes (se recorre recursivamente).
        batch_size: Cantidad de imágenes por batch.
        image_size: Lado del cuadrado de salida en píxeles.
        augment: Si es ``True``, aplica volteo horizontal aleatorio.
        crop: Modo de encuadre (ver :func:`_build_transform`).
        num_workers: Procesos de carga del ``DataLoader``.
        shuffle: Si es ``True``, baraja en cada recorrido.
        seed: Semilla del barajado. Con ``seed`` fijo, ``augment=False`` y
            ``num_workers=0`` la secuencia es reproducible. ``None`` usa el RNG
            global.
        pin_memory: Se pasa tal cual al ``DataLoader``.

    Raises:
        ValueError: Si ``root`` no existe, no contiene imágenes o contiene menos
            de ``batch_size`` (con ``drop_last=True`` el loader quedaría vacío y
            el iterador infinito se colgaría). Se levanta en la llamada.
    """
    import torch

    transform = _build_transform(image_size, augment, crop)
    dataset = _build_cat_images_class()(root, transform)

    n_images = len(dataset)
    if n_images < batch_size:
        raise ValueError(
            f"La carpeta {Path(root)!s} tiene {n_images} imagen(es), menos que "
            f"batch_size={batch_size}: con drop_last=True el loader quedaría "
            f"vacío y el iterador infinito no entregaría ningún batch."
        )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    return _infinite(loader)


def finite_batches(
    root: str | Path,
    batch_size: int,
    *,
    image_size: int = 64,
    crop: bool = True,
    max_images: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Iterable[torch.Tensor]:
    """Fuente finita y re-iterable de batches de imágenes, en orden fijo, para evaluar.

    Mismo preprocesamiento que :func:`infinite_batches` pero sin augmentation,
    sin barajar y con ``drop_last=False`` (el último batch puede ser parcial).
    No consume RNG global, y recorrerla dos veces entrega los mismos batches.

    Args:
        root: Carpeta raíz con las imágenes (se recorre recursivamente).
        batch_size: Cantidad de imágenes por batch.
        image_size: Lado del cuadrado de salida en píxeles.
        crop: Modo de encuadre (ver :func:`_build_transform`).
        max_images: Si se da, usa solo las primeras N imágenes en orden de
            descubrimiento (un tope mayor que el set no recorta). ``None`` usa
            todas.
        num_workers: Procesos de carga del ``DataLoader``.
        pin_memory: Se pasa tal cual al ``DataLoader``.

    Returns:
        Iterable de tensores ``(b, 3, image_size, image_size)`` float32 en
        ``[-1, 1]``: ``ceil(n / batch_size)`` batches que suman
        ``n = min(count_images(root), max_images)``.

    Raises:
        ValueError: Si ``root`` no existe o no contiene imágenes.
    """
    import torch

    transform = _build_transform(image_size, augment=False, crop=crop)
    dataset = _build_cat_images_class()(root, transform)

    if max_images is not None:
        n_taken = min(max_images, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(n_taken))

    # Sin generator propio, el DataLoader sortea su base_seed del RNG global
    # (aun con shuffle=False y num_workers=0) y adelantaría el azar del proceso.
    generator = torch.Generator()
    generator.manual_seed(0)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def count_images(root: str | Path) -> int:
    """Cuenta las imágenes bajo ``root`` con el mismo criterio que la carga.

    Útil para dimensionar un recorrido, p. ej.
    ``finite_batches(..., max_images=count_images(otra_carpeta))``.

    Raises:
        ValueError: Si ``root`` no existe o no contiene ninguna imagen.
    """
    return len(_discover_image_paths(root))


def report_small_images(
    root: str | Path,
    *,
    min_size: int = 64,
    verbose: bool = False,
) -> list[Path]:
    """Devuelve (sin borrar nada) las imágenes cuyo lado corto es menor que ``min_size``.

    Esas imágenes se escalarían hacia arriba en el ``Resize`` y entrarían
    borrosas. Solo lee las dimensiones; no modifica archivos.

    Args:
        root: Carpeta raíz con las imágenes (se recorre recursivamente).
        min_size: Umbral en píxeles para ``min(width, height)``.
        verbose: Si es ``True``, imprime cada ruta reportada con sus dimensiones.

    Raises:
        ValueError: Si ``root`` no existe o no contiene ninguna imagen.
    """
    from PIL import Image

    small: list[Path] = []
    for path in _discover_image_paths(root):
        with Image.open(path) as img:
            width, height = img.size
        if min(width, height) < min_size:
            small.append(path)
            if verbose:
                print(f"  too-small: {path.name} ({width}x{height})")
    return small


def __getattr__(name: str):
    """Resuelve ``CatImages`` de forma perezosa (PEP 562) y la cachea en el módulo."""
    if name == "CatImages":
        cls = _build_cat_images_class()
        globals()["CatImages"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    # Smoke test manual sobre data/cats-prueba/:
    #   python -m diffusion.data_generation.images
    import torch

    # parents[3] es la raíz del repo.
    cats_prueba = Path(__file__).resolve().parents[3] / "data" / "cats-prueba"

    batch = next(infinite_batches(cats_prueba, batch_size=2, image_size=64))
    assert tuple(batch.shape) == (2, 3, 64, 64), f"shape inesperada: {tuple(batch.shape)}"
    assert batch.dtype == torch.float32, f"dtype inesperado: {batch.dtype}"
    lo, hi = float(batch.min()), float(batch.max())
    assert lo >= -1.01 and hi <= 1.01, f"rango fuera de ~[-1, 1]: [{lo:.3f}, {hi:.3f}]"
    print(
        f"infinite_batches(cats-prueba, batch_size=2): batch {tuple(batch.shape)} "
        f"{batch.dtype}, rango [{lo:.3f}, {hi:.3f}]"
    )

    small = report_small_images(cats_prueba, min_size=64)
    print(
        f"report_small_images(min_size=64): {len(small)} imagen(es) demasiado "
        f"chica(s) (report-only, no se borra ninguna)"
    )
