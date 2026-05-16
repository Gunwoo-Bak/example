# CrackSeg Repository

RGB images are segmented into damage classes with the Dachung crack segmentation model. The repository is built from the useful source code in the existing `CrackSegmentation` project, while excluding PyInstaller `build/` and `dist/` artifacts.

## What This Repository Does

- Runs semantic segmentation on one RGB image or an image folder.
- Saves color masks, overlay visualizations, and confidence maps.
- Computes pixel-level prediction statistics by class.
- Extracts connected components per predicted damage class for instance-style summaries.

There is no ground-truth mask in the current data. Because of that, this repository cannot compute true accuracy, IoU, F1, precision, or recall against labels. The quantitative outputs are prediction-based statistics: pixel counts, class ratios, and detected component summaries.

## Classes

The default config uses the Dachung classes:

| ID | Class |
| --- | --- |
| 0 | BG |
| 1 | CRC |
| 2 | DLM |
| 3 | SPL |

## Repository Layout

```text
crackseg-repository/
  configs/
    dachung.yaml              # model, class, tile, and palette settings
  scripts/
    predict.py                # run one image
    batch_predict.py          # run a folder of images
  src/crackseg/
    config.py                 # YAML config loading
    models.py                 # segmentation model factory
    inference.py              # checkpoint loading and sliding-window inference
    visualization.py          # color mask and overlay rendering
    metrics.py                # pixel-level prediction summaries
    instances.py              # connected-component summaries
    export.py                 # output file writing
    io.py                     # image reading and discovery
  weights/
    README.md                 # checkpoint placement note
  tests/
```

## Setup

```bash
cd /home/gunwoo/CrackSegmentation/crackseg-repository
python -m pip install -r requirements.txt
python -m pip install -e .
```

The known local Dachung checkpoint is:

```text
/home/gunwoo/CrackSegmentation/outputs/dachung_finetune/best.pt
```

It is not copied into the repository because it is about 175 MB. Pass it with `--checkpoint`, or place your own checkpoint under `weights/`.

## Single Image Inference

```bash
cd /home/gunwoo/CrackSegmentation/crackseg-repository
python scripts/predict.py \
  --image /path/to/input_rgb.png \
  --checkpoint /home/gunwoo/CrackSegmentation/outputs/dachung_finetune/best.pt \
  --config configs/dachung.yaml \
  --output-dir outputs/predictions
```

Outputs:

```text
outputs/predictions/
  masks/<image>_mask.png
  overlays/<image>_overlay.png
  confidence/<image>_confidence.png
  summaries/<image>_pixel_summary.csv
  summaries/<image>_metadata.json
  instances/<image>_instances.csv
  instances/<image>_instance_summary.csv
```

## Batch Inference

```bash
cd /home/gunwoo/CrackSegmentation/crackseg-repository
python scripts/batch_predict.py \
  --input-dir /path/to/rgb_images \
  --checkpoint /home/gunwoo/CrackSegmentation/outputs/dachung_finetune/best.pt \
  --config configs/dachung.yaml \
  --output-dir outputs/batch_predictions
```

Batch mode additionally saves:

```text
outputs/batch_predictions/batch_pixel_summary.csv
```

## Pixel-Level Quantification Without GT

For each predicted mask, the repository reports:

- `pixel_count`: number of pixels assigned to each class.
- `pixel_ratio`: class pixel count divided by total image pixels.
- `area_percent`: `pixel_ratio * 100`.

These values measure the model output distribution, not correctness. To evaluate correctness, prepare GT masks with the same class IDs and add a GT evaluation script for IoU/F1.

## Tile Inference

Large images are processed by sliding-window inference. Defaults are controlled in `configs/dachung.yaml`:

```yaml
inference:
  img_size: 768
  stride: 384
```

Use CLI overrides when needed:

```bash
python scripts/predict.py ... --tile-size 512 --stride 256
```
