#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crack/damage segmentation for one RGB image.")
    parser.add_argument("--image", required=True, help="Input RGB image path.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path, for example outputs/dachung_finetune/best.pt.")
    parser.add_argument("--config", default="configs/dachung.yaml", help="Config YAML path.")
    parser.add_argument("--output-dir", default="outputs/predictions", help="Directory for prediction outputs.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or omitted for automatic selection.")
    parser.add_argument("--tile-size", type=int, default=None, help="Override inference tile size.")
    parser.add_argument("--stride", type=int, default=None, help="Override inference stride.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for mask color.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from crackseg.export import save_prediction_outputs
    from crackseg.inference import load_model_bundle, sliding_window_inference
    from crackseg.io import read_image_rgb

    bundle = load_model_bundle(
        checkpoint=args.checkpoint,
        config_path=args.config,
        device=args.device,
        tile_size=args.tile_size,
        stride=args.stride,
    )
    image = read_image_rgb(args.image)
    pred_mask, confidence, _ = sliding_window_inference(image, bundle)
    stem = Path(args.image).stem
    paths = save_prediction_outputs(
        image=image,
        pred_mask=pred_mask,
        confidence=confidence,
        output_dir=args.output_dir,
        stem=stem,
        config=bundle.config,
        alpha=args.alpha,
    )
    print("Saved prediction outputs:")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
