#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crack/damage segmentation for an image directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing RGB images.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument("--config", default="configs/dachung.yaml", help="Config YAML path.")
    parser.add_argument("--output-dir", default="outputs/batch_predictions", help="Directory for prediction outputs.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or omitted for automatic selection.")
    parser.add_argument("--tile-size", type=int, default=None, help="Override inference tile size.")
    parser.add_argument("--stride", type=int, default=None, help="Override inference stride.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for mask color.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from tqdm import tqdm

    from crackseg.export import save_prediction_outputs, write_batch_summary
    from crackseg.inference import load_model_bundle, sliding_window_inference
    from crackseg.io import list_images, read_image_rgb
    from crackseg.metrics import summarize_pixels

    image_paths = list_images(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    bundle = load_model_bundle(
        checkpoint=args.checkpoint,
        config_path=args.config,
        device=args.device,
        tile_size=args.tile_size,
        stride=args.stride,
    )

    batch_rows = []
    for image_path in tqdm(image_paths, desc="Predicting"):
        image = read_image_rgb(image_path)
        pred_mask, confidence, _ = sliding_window_inference(image, bundle)
        save_prediction_outputs(
            image=image,
            pred_mask=pred_mask,
            confidence=confidence,
            output_dir=args.output_dir,
            stem=image_path.stem,
            config=bundle.config,
            alpha=args.alpha,
            source_path=image_path,
        )
        for row in summarize_pixels(pred_mask, bundle.config.class_names):
            batch_rows.append({"image": image_path.name, **row})

    summary_path = Path(args.output_dir) / "batch_pixel_summary.csv"
    write_batch_summary(batch_rows, summary_path)
    print(f"Processed images: {len(image_paths)}")
    print(f"Batch pixel summary: {summary_path}")


if __name__ == "__main__":
    main()
