from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .config import CrackSegConfig, load_config
from .models import create_model


@dataclass
class ModelBundle:
    model: torch.nn.Module
    device: torch.device
    config: CrackSegConfig


def preprocess_for_model(rgb: np.ndarray) -> torch.Tensor:
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def make_positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def pad_to_tile(crop: np.ndarray, tile: int) -> np.ndarray:
    h, w = crop.shape[:2]
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    if pad_h == 0 and pad_w == 0:
        return crop
    return np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def load_model_bundle(
    checkpoint: str | Path,
    config_path: str | Path = "configs/dachung.yaml",
    device: str | None = None,
    tile_size: int | None = None,
    stride: int | None = None,
) -> ModelBundle:
    config = load_config(config_path)
    if tile_size is not None or stride is not None:
        config = CrackSegConfig(
            classes=config.classes,
            num_classes=config.num_classes,
            model=config.model,
            encoder_name=config.encoder_name,
            img_size=int(tile_size or config.img_size),
            stride=int(stride or config.stride),
            palette=config.palette,
        )

    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_model(
        config.model,
        config.encoder_name,
        config.num_classes,
        encoder_weights=None,
    ).to(run_device)

    ckpt = torch.load(checkpoint, map_location=run_device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return ModelBundle(model=model, device=run_device, config=config)


@torch.no_grad()
def sliding_window_inference(
    image_rgb: np.ndarray,
    bundle: ModelBundle,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    cfg = bundle.config
    h, w = image_rgb.shape[:2]
    y_positions = make_positions(h, cfg.img_size, cfg.stride)
    x_positions = make_positions(w, cfg.img_size, cfg.stride)

    logits_sum = np.zeros((h, w, cfg.num_classes), dtype=np.float32)
    counts = np.zeros((h, w, 1), dtype=np.float32)

    total_tiles = len(y_positions) * len(x_positions)
    tile_index = 0

    for y0 in y_positions:
        for x0 in x_positions:
            y1 = min(y0 + cfg.img_size, h)
            x1 = min(x0 + cfg.img_size, w)

            crop = image_rgb[y0:y1, x0:x1]
            crop_pad = pad_to_tile(crop, cfg.img_size)
            inp = preprocess_for_model(crop_pad).unsqueeze(0).to(bundle.device)

            logits = bundle.model(inp)
            logits_np = logits.squeeze(0).detach().cpu().numpy()
            logits_np = logits_np[:, : y1 - y0, : x1 - x0]
            logits_np = np.transpose(logits_np, (1, 2, 0))

            logits_sum[y0:y1, x0:x1] += logits_np
            counts[y0:y1, x0:x1] += 1.0

            tile_index += 1
            if progress_callback is not None:
                progress_callback(tile_index, total_tiles)

    avg_logits = logits_sum / np.clip(counts, 1e-6, None)
    pred_mask = np.argmax(avg_logits, axis=2).astype(np.uint8)
    confidence_map = torch.softmax(torch.from_numpy(avg_logits), dim=-1).numpy().max(axis=-1)
    return pred_mask, confidence_map, total_tiles
