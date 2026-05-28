# CrackSeg

Semantic segmentation inference for concrete surface damage images. The model predicts three damage classes from RGB images or TIFF/GeoTIFF files:

| ID | Class | Meaning | Overlay Color |
| ---: | --- | --- | --- |
| 0 | BG | background | black |
| 1 | CRC | crack / 균열 | red |
| 2 | DLM | delamination / 박리 | green |
| 3 | SPL | spalling / 박락 | orange |

The repository is inference-focused. It does not include training data, GT masks, model weights, or generated outputs.
Download the model checkpoint separately from the repository release assets before running inference.

## Features

- Single-image and batch segmentation.
- RGB image and TIFF input support.
- Full prediction outputs: mask, overlay, confidence map, pixel summary, instance summary.
- Damage-type outputs separated into `CRC`, `DLM`, and `SPL` folders.
- GeoTIFF scale/tiepoint support for local meter coordinates and `m2` area summaries.
- Prediction-based quantification only. Accuracy, IoU, F1, precision, and recall require GT masks.

## Repository Layout

```text
crackseg-repository/
  configs/
    dachung.yaml
  scripts/
    predict.py
    batch_predict.py
  src/crackseg/
    config.py
    export.py
    geotiff.py
    inference.py
    instances.py
    io.py
    metrics.py
    models.py
    visualization.py
  tests/
    test_metrics.py
  weights/
    README.md
    best.pt       # downloaded separately from GitHub Releases
```

Ignored local folders:

```text
data/       input images, not committed
outputs/    generated results, not committed
weights/*.pt
```

## Setup

```bash
cd /home/gunwoo/CrackSegmentation/crackseg-repository
conda create -n crackseg python=3.10 -y
conda activate crackseg

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Check CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Model Weights

Model checkpoints are intentionally excluded from Git because they are large. Download `best.pt` from the repository's GitHub Release assets and place it under `weights/`.

Expected local layout:

```text
crackseg-repository/
  weights/
    best.pt
```

If the Release page provides a direct download URL, you can run:

```bash
cd /home/gunwoo/CrackSegmentation/crackseg-repository
mkdir -p weights
wget -O weights/best.pt "https://github.com/Gunwoo-Bak/example/releases/download/v1.0.0/best.pt"
```

Alternatively, download `best.pt` manually from:

```text
https://github.com/<OWNER>/<REPOSITORY>/releases
```

and move it to:

```text
weights/best.pt
```

You can also pass an absolute checkpoint path with `--checkpoint`.

Known local checkpoint:

```text
/home/gunwoo/CrackSegmentation/outputs/dachung_finetune/best.pt
```

For maintainers creating a release, attach this file as the binary asset:

```text
/home/gunwoo/CrackSegmentation/outputs/dachung_finetune/best.pt
```

## Single Image Inference

```bash
cd /home/gunwoo/CrackSegmentation/crackseg-repository
conda activate crackseg

python scripts/predict.py \
  --image data/daechung_298.tif \
  --checkpoint weights/best.pt \
  --device cuda \
  --output-dir outputs/daechung_298_test
```

## Batch Inference

```bash
cd /home/gunwoo/CrackSegmentation/crackseg-repository
conda activate crackseg

python scripts/batch_predict.py \
  --input-dir data \
  --checkpoint weights/best.pt \
  --device cuda \
  --output-dir outputs/predictions
```

Supported input extensions:

```text
.jpg .jpeg .png .tif .tiff
```

## Output Structure

For input `daechung_298.tif`, outputs are saved like this:

```text
outputs/daechung_298_test/
  masks/
    daechung_298_mask.png
  overlays/
    daechung_298_overlay.png
  confidence/
    daechung_298_confidence.png
  summaries/
    daechung_298_pixel_summary.csv
    daechung_298_metadata.json
  instances/
    daechung_298_instances.csv
    daechung_298_instance_summary.csv
  by_damage/
    01_CRC/
      masks/
      overlays/
      summaries/
      instances/
    02_DLM/
      masks/
      overlays/
      summaries/
      instances/
    03_SPL/
      masks/
      overlays/
      summaries/
      instances/
```

Damage-specific examples:

```text
by_damage/01_CRC/masks/daechung_298_CRC_binary_mask.png
by_damage/01_CRC/overlays/daechung_298_CRC_overlay.png
by_damage/01_CRC/summaries/daechung_298_CRC_pixel_summary.csv
by_damage/01_CRC/instances/daechung_298_CRC_instances.csv
```

## Quantitative Outputs

`summaries/<image>_pixel_summary.csv` contains:

```text
class_id
class_name
pixel_count
pixel_ratio
area_percent
area_m2              # only when GeoTIFF pixel scale exists
pixel_size_x_m       # only when GeoTIFF pixel scale exists
pixel_size_y_m       # only when GeoTIFF pixel scale exists
```

`instances/<image>_instances.csv` contains connected-component summaries:

```text
instance_id
class_id
class_name
centroid_x_px
centroid_y_px
bbox_xmin_px
bbox_ymin_px
bbox_xmax_px
bbox_ymax_px
area_px
centroid_x_m         # only when GeoTIFF pixel scale/tiepoint exists
centroid_y_m
bbox_xmin_m
bbox_ymin_m
bbox_xmax_m
bbox_ymax_m
area_m2
```

These are prediction statistics, not GT-based accuracy metrics.

## GeoTIFF Notes

If a TIFF contains `ModelPixelScaleTag` and `ModelTiepointTag`, the repository adds local meter-scale fields to CSV outputs.

If CRS text is `Arbitrary (m)`, the coordinates are local meter coordinates, not longitude/latitude or EPSG coordinates.

## Inference Tiling

Defaults are in `configs/dachung.yaml`:

```yaml
img_size: 768
stride: 384
```

Override at runtime:

```bash
python scripts/predict.py \
  --image data/daechung_298.tif \
  --checkpoint weights/best.pt \
  --tile-size 512 \
  --stride 256
```

## Tests

```bash
cd /home/gunwoo/CrackSegmentation/crackseg-repository
python -m pytest -q
```
