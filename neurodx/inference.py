"""Load a trained checkpoint and run calibrated, explainable predictions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

from .config import IMAGENET_MEAN, IMAGENET_STD, NUM_SEG_CLASSES
from .models import build_model


@dataclass
class LoadedModel:
    model: torch.nn.Module
    classes: list[str]
    image_size: int
    temperature: float
    device: torch.device
    meta: dict = field(default_factory=dict)


def load_checkpoint(path: str | Path, device: str | torch.device | None = None) -> LoadedModel:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(path, map_location=device, weights_only=True)
    args = ckpt["args"]
    model = build_model(args["arch"], args["backbone"], len(ckpt["classes"]), NUM_SEG_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return LoadedModel(model, list(ckpt["classes"]), int(args.get("image_size", 224)),
                       float(ckpt.get("temperature", 1.0)), device,
                       {"arch": args["arch"], "backbone": args["backbone"], "disease": args.get("disease"),
                        "epoch": ckpt.get("epoch")})


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB")
    image = TF.resize(image, [image_size, image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0)


def grad_cam(model: torch.nn.Module, x: torch.Tensor, target: int | None = None) -> tuple[np.ndarray, torch.Tensor]:
    """Grad-CAM (Selvaraju et al., 2017) on the deepest encoder stage.
    Returns (heatmap in [0,1] of shape HxW, class logits)."""
    acts: dict[str, torch.Tensor] = {}
    layer = model.encoder.stages[-1]
    handle = layer.register_forward_hook(lambda _m, _i, out: acts.__setitem__("a", out))
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            _, logits = model(x)
            a = acts["a"]
            a.retain_grad()
            idx = int(logits.argmax(1)) if target is None else target
            logits[0, idx].backward()
            weights = a.grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * a).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    finally:
        handle.remove()
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, logits.detach()


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-coloured heatmap over the (resized) grayscale image."""
    import matplotlib

    base = np.asarray(image.convert("L").resize(cam.shape[::-1]), np.float32) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    rgb = (1 - alpha) * np.repeat(base[..., None], 3, axis=2) + alpha * heat
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def predict(lm: LoadedModel, image: Image.Image, abstain_below: float = 0.0, explain: bool = False) -> dict:
    """Calibrated class probabilities for one slice. ``abstained`` is True when
    the top probability is below ``abstain_below`` (selective prediction)."""
    x = preprocess(image, lm.image_size).to(lm.device)
    if explain:
        cam, logits = grad_cam(lm.model, x)
    else:
        with torch.no_grad():
            _, logits = lm.model(x)
        cam = None
    probs = torch.softmax(logits.float() / lm.temperature, 1)[0].cpu().numpy()
    top = int(probs.argmax())
    out = {
        "prediction": lm.classes[top],
        "confidence": float(probs[top]),
        "probabilities": {c: float(p) for c, p in zip(lm.classes, probs)},
        "abstained": bool(probs[top] < abstain_below),
        "temperature": lm.temperature,
    }
    if cam is not None:
        out["cam"] = cam
    return out
