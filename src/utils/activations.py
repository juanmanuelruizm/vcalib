"""Utilities for loading RF-DETR (via LibreYOLO) and extracting/caching activations.

The frozen RF-DETR nano model is wrapped by LibreYOLO (`LibreRFDETR`). The internal
detection model is an `LWDETR` nn.Module reachable as `libre.model.model`. Because
LibreYOLO does not expose `output_hidden_states` through its wrapper, intermediate
activations are captured with PyTorch `register_forward_hook` on selected submodules
(DINOv2 backbone blocks, multi-scale projector, decoder layers).

Activation capture happens during a normal inference forward pass: the model runs in
`eval()` under `torch.no_grad()` and each hook copies a detached output tensor to CPU.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union, cast

import numpy as np
import torch
from PIL import Image

ImageLike = Union[str, Path, Image.Image, np.ndarray, torch.Tensor]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "3rd_party" / "libreyolo" / "weights"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "processed" / "activations_cache"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INPUT_SIZES: Dict[str, int] = {"n": 384, "s": 512, "m": 576, "l": 704}

LAYER_PATHS: Dict[str, str] = {
    **{f"backbone.layer.{i}": f"backbone.0.encoder.encoder.encoder.layer.{i}" for i in range(12)},
    "backbone.projector": "backbone.0.projector",
    "decoder.layer.0": "transformer.decoder.layers.0",
    "decoder.layer.1": "transformer.decoder.layers.1",
}
DEFAULT_LAYERS: Tuple[str, ...] = tuple(LAYER_PATHS.keys())

_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(
    size: str = "n",
    device: Optional[str] = None,
    weights_dir: Optional[Union[str, Path]] = None,
) -> Any:
    """Load a frozen RF-DETR model via LibreYOLO.

    Args:
        size: Model size code ("n"=nano, "s", "m", "l").
        device: "auto" (default), "cuda", or "cpu".
        weights_dir: Directory holding `<size>` weights. Defaults to the
            submodule's ``weights/`` dir; ``rf-detr-nano.pth`` is auto-downloaded
            there on first use.

    Returns:
        The ``LibreRFDETR`` wrapper (cached per size/device/weights_dir).
    """
    from libreyolo import LibreRFDETR

    if size not in INPUT_SIZES:
        raise ValueError(f"Unknown size {size!r}; expected one of {list(INPUT_SIZES)}")

    wdir = Path(weights_dir).resolve() if weights_dir else DEFAULT_WEIGHTS_DIR.resolve()
    wdir.mkdir(parents=True, exist_ok=True)
    pretrain = {
        "n": "rf-detr-nano.pth",
        "s": "rf-detr-small.pth",
        "m": "rf-detr-medium.pth",
        "l": "rf-detr-large.pth",
    }[size]
    model_path = wdir / pretrain

    dev = _resolve_device(device)
    cache_key = (size, str(dev), str(wdir))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    libre = LibreRFDETR(model_path=str(model_path), size=size, device=str(dev))
    libre.model.eval()
    for p in libre.model.parameters():
        p.requires_grad_(False)
    _MODEL_CACHE[cache_key] = libre
    return libre


def _inner_model(libre: Any) -> torch.nn.Module:
    return cast(torch.nn.Module, libre.model.model)


def _model_device(libre: Any) -> torch.device:
    return cast(torch.device, next(libre.model.parameters()).device)


def load_image(image: ImageLike) -> Image.Image:
    """Load an image from a path/PIL/array/tensor into a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path)):
        return Image.open(image)
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().float()
        if arr.dim() == 4:
            arr = arr[0]
        if arr.dim() == 3 and arr.shape[0] in (1, 3) and arr.shape[0] <= arr.shape[2]:
            arr = arr.permute(1, 2, 0)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return Image.fromarray(arr.numpy().astype(np.uint8))
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise TypeError(f"Unsupported image type: {type(image)}")


def to_unit_rgb(image: ImageLike, input_size: Optional[int] = None) -> torch.Tensor:
    """Return an RGB image as a (3, H, W) float tensor in [0, 1].

    This is the filter insertion point: a parametric calibration filter operates on
    this [0, 1] tensor before :func:`normalize` applies ImageNet statistics.
    """
    size = input_size if input_size is not None else INPUT_SIZES["n"]
    img = load_image(image).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def normalize(unit_tensor: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std normalization to a (C, H, W) [0, 1] tensor."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=unit_tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=unit_tensor.dtype).view(3, 1, 1)
    return (unit_tensor - mean) / std


def preprocess(image: ImageLike, size: str = "n", input_size: Optional[int] = None) -> torch.Tensor:
    """Preprocess an image into a normalized (1, 3, H, W) model-input tensor.

    Mirrors LibreYOLO's RF-DETR preprocessor: RGB, bilinear resize to a square
    input, /255, then ImageNet mean/std normalization, CHW float32.
    """
    res = input_size if input_size is not None else INPUT_SIZES[size]
    unit = to_unit_rgb(image, res)
    return normalize(unit).unsqueeze(0)


def resolve_layer_path(name: str) -> str:
    """Map a canonical layer name to its dotted submodule path (pass-through if already full)."""
    if name in LAYER_PATHS:
        return LAYER_PATHS[name]
    return name


def _to_tensor(output: Any) -> Optional[torch.Tensor]:
    """Reduce a module output to a single representative tensor, if possible."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state
    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        hs = output.hidden_states
        if isinstance(hs, (list, tuple)) and hs and isinstance(hs[-1], torch.Tensor):
            return hs[-1]
    if isinstance(output, (list, tuple)):
        for elem in output:
            t = _to_tensor(elem)
            if t is not None:
                return t
    return None


class ActivationExtractor:
    """Capture intermediate activations of RF-DETR via forward hooks during inference.

    Hooks are registered on submodules identified by canonical layer names (see
    :data:`LAYER_PATHS`). The extractor is a context manager; hooks are removed on
    exit or via :meth:`remove_hooks`.
    """

    def __init__(
        self,
        model: Any,
        layers: Iterable[str] = DEFAULT_LAYERS,
    ) -> None:
        self.libre = model
        self.inner = _inner_model(model)
        self.layers: Tuple[str, ...] = tuple(layers)
        self._handles: list = []
        self._acts: Dict[str, torch.Tensor] = {}
        self.last_output: Any = None
        self._register()

    def _resolve_path(self, name: str) -> str:
        return resolve_layer_path(name)

    def _register(self) -> None:
        for name in self.layers:
            path = self._resolve_path(name)
            try:
                module = self.inner.get_submodule(path)
            except AttributeError as exc:
                raise AttributeError(f"Layer {name!r} (path {path!r}) not found in model") from exc
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name: str):
        def _hook(_module: torch.nn.Module, _input: Any, output: Any) -> None:
            try:
                t = _to_tensor(output)
                if t is not None:
                    self._acts[name] = t.detach().to("cpu")
            except Exception:
                pass

        return _hook

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def extract(self, image: ImageLike) -> Dict[str, torch.Tensor]:
        """Run a frozen inference forward pass and return captured activations."""
        from_size = getattr(self.libre, "size", "n")
        tensor = preprocess(image, size=from_size)
        tensor = tensor.to(_model_device(self.libre))
        self._acts.clear()
        self.libre.model.eval()
        with torch.no_grad():
            self.last_output = self.libre.model(tensor)
        return dict(self._acts)

    def __enter__(self) -> "ActivationExtractor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove_hooks()


def extract_activations(
    image: ImageLike,
    layers: Iterable[str] = DEFAULT_LAYERS,
    size: str = "n",
    device: Optional[str] = None,
    weights_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, torch.Tensor]:
    """Convenience: load (cached) model, run inference, return activations dict."""
    libre = load_model(size=size, device=device, weights_dir=weights_dir)
    with ActivationExtractor(libre, layers=layers) as ex:
        return ex.extract(image)


def list_available_layers(
    size: str = "n",
    device: Optional[str] = None,
    weights_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Return canonical layer names mapped to their nn.Module class names."""
    libre = load_model(size=size, device=device, weights_dir=weights_dir)
    inner = _inner_model(libre)
    out: Dict[str, str] = {}
    for canon, path in LAYER_PATHS.items():
        try:
            mod = inner.get_submodule(path)
            out[canon] = type(mod).__name__
        except AttributeError:
            out[canon] = "<missing>"
    return out


def cache_activations(
    activations: Mapping[str, torch.Tensor],
    name: str,
    cache_dir: Optional[Union[str, Path]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save an activations dict to ``<cache_dir>/<name>.pt`` (+ optional ``.meta.json``)."""
    cdir = Path(cache_dir).resolve() if cache_dir else DEFAULT_CACHE_DIR.resolve()
    cdir.mkdir(parents=True, exist_ok=True)
    pt_path = cdir / f"{name}.pt"
    torch.save(dict(activations), pt_path)
    if metadata is not None:
        meta_path = cdir / f"{name}.meta.json"
        meta = dict(metadata)
        meta.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        meta["activations"] = {
            k: list(v.shape) for k, v in activations.items() if isinstance(v, torch.Tensor)
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return pt_path


def load_cached_activations(
    name: str,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, torch.Tensor]:
    """Load activations previously saved by :func:`cache_activations`."""
    cdir = Path(cache_dir).resolve() if cache_dir else DEFAULT_CACHE_DIR.resolve()
    path = cdir / f"{name}.pt"
    return cast(Dict[str, torch.Tensor], torch.load(path, map_location="cpu", weights_only=True))


def _synthetic_image(seed: int = 42) -> np.ndarray:
    """Deterministic RGB gradient image for smoke tests (no dataset required)."""
    rng = np.random.default_rng(seed)
    size = INPUT_SIZES["n"]
    xs: np.ndarray = np.linspace(0, 1, size, dtype=np.float32)
    ys: np.ndarray = np.linspace(0, 1, size, dtype=np.float32)
    grad = np.outer(ys, xs)
    noise = rng.normal(0, 0.03, (size, size)).astype(np.float32)
    base = np.clip(grad + noise, 0, 1)
    img = np.stack(
        [base, np.clip(base * 0.8 + 0.1, 0, 1), np.clip(1.0 - base, 0, 1)],
        axis=-1,
    )
    return cast(np.ndarray, (img * 255).astype(np.uint8))


def _run_test(
    image_path: Optional[str],
    save_name: Optional[str],
    cache_dir: Optional[str],
    size: str,
    device: Optional[str],
) -> None:
    image: ImageLike = image_path if image_path else _synthetic_image()
    print(f"[activations] size={size} device={_resolve_device(device)}")
    print("[activations] loading model (frozen) ...")
    libre = load_model(size=size, device=device)

    print("[activations] available layers:")
    for canon, cls in list_available_layers(size=size, device=device).items():
        print(f"  {canon:22s} {cls}")

    print("[activations] running inference forward pass with hooks ...")
    with ActivationExtractor(libre, layers=DEFAULT_LAYERS) as ex:
        acts = ex.extract(image)
        out = ex.last_output

    print(f"[activations] captured {len(acts)} layers:")
    for name in DEFAULT_LAYERS:
        if name in acts:
            print(f"  {name:22s} shape={tuple(acts[name].shape)} dtype={acts[name].dtype}")

    if isinstance(out, dict):
        print("[activations] detection output keys:")
        for k in ("pred_logits", "pred_boxes"):
            if k in out and isinstance(out[k], torch.Tensor):
                t = out[k].detach().to("cpu")
                print(f"  {k:22s} shape={tuple(t.shape)}")

    if save_name:
        meta = {
            "model": f"rf-detr-{size}",
            "source": image_path or "synthetic",
            "layers": list(DEFAULT_LAYERS),
        }
        path = cache_activations(acts, save_name, cache_dir=cache_dir, metadata=meta)
        print(f"[activations] saved -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-DETR activation extraction (LibreYOLO).")
    parser.add_argument(
        "--test", action="store_true", help="run a smoke extraction and print shapes"
    )
    parser.add_argument("--list-layers", action="store_true", help="print the layer-name map")
    parser.add_argument(
        "--image", type=str, default=None, help="image path for --test (else synthetic)"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="cache extracted activations under this name"
    )
    parser.add_argument("--size", type=str, default="n", help="model size: n, s, m, l")
    parser.add_argument("--device", type=str, default="auto", help="cuda | cpu | auto")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="activations cache directory",
    )
    args = parser.parse_args()

    if args.list_layers:
        for canon, cls in list_available_layers(size=args.size, device=args.device).items():
            print(f"{canon:22s} {cls}")
        return

    if args.test:
        _run_test(args.image, args.save, args.cache_dir, args.size, args.device)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
