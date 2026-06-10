# CrackSeg

콘크리트 표면 RGB 이미지에 대한 손상 segmentation 모델 파이프라인입니다. RGB 이미지 또는 TIFF/GeoTIFF 파일을 입력으로 받아 아래 세 가지 손상 클래스를 예측합니다.

| ID | Class | 의미 | Overlay 색상 |
| ---: | --- | --- | --- |
| 0 | BG | 배경 | black |
| 1 | CRC | crack / 균열 | red |
| 2 | DLM | delamination / 박리 | green |
| 3 | SPL | spalling / 박락 | orange |

모델을 실행하기 전에 GitHub Release asset에서 모델 checkpoint(best.pt)를 별도로 다운로드해야 합니다.

## 주요 기능

- 단일 이미지 및 폴더 단위 batch segmentation 지원
- RGB 이미지와 TIFF/GeoTIFF 입력 지원
- mask, overlay, confidence map, pixel summary, instance summary 출력
- 균열, 박리, 박락 손상 유형별 결과 폴더 분리 저장
- GeoTIFF의 scale/tiepoint 정보를 이용한 local meter 좌표 및 산출 면적 지원
- 모델 예측 결과 기반 정량 결과 산출

## Repository 구조

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
    best.pt       # GitHub Releases에서 별도 다운로드
```

주요 파일과 폴더의 역할은 다음과 같습니다.

| 경로 | 설명 |
| --- | --- |
| `configs/dachung.yaml` | 탐지에 사용하는 기본 설정 파일입니다. 클래스 이름, 모델 구조, tile 크기, stride, overlay 색상 팔레트가 정의되어 있습니다. |
| `scripts/predict.py` | 단일 이미지 segmentation을 수행하는 실행 스크립트입니다. |
| `scripts/batch_predict.py` | 입력 폴더 안의 여러 이미지를 한 번에 처리하는 batch 실행 스크립트입니다. |
| `src/crackseg/config.py` | YAML 설정을 읽고 탐지 설정 객체로 변환합니다. |
| `src/crackseg/models.py` | segmentation 모델 구조를 생성합니다. |
| `src/crackseg/inference.py` | checkpoint 로드와 sliding-window 기반 추론을 수행합니다. |
| `src/crackseg/io.py` | RGB 이미지, TIFF/GeoTIFF 입력 파일을 읽고 batch 대상 이미지 목록을 구성합니다. |
| `src/crackseg/export.py` | mask, overlay, confidence map, summary CSV, metadata JSON 등 결과 파일을 저장합니다. |
| `src/crackseg/geotiff.py` | GeoTIFF scale/tiepoint 정보를 읽어 pixel 좌표를 local meter 좌표로 변환합니다. |
| `src/crackseg/metrics.py` | 클래스별 pixel 수, pixel 비율, 손상율, 면적 등 정량 지표를 계산합니다. |
| `src/crackseg/instances.py` | connected component 기반으로 손상 instance를 분리하고 중심점, bounding box, 면적을 계산합니다. |
| `src/crackseg/visualization.py` | 예측 mask를 색상 overlay와 confidence map으로 시각화합니다. |
| `tests/test_metrics.py` | 정량 지표 계산 로직에 대한 테스트 코드입니다. |
| `weights/` | 모델 checkpoint를 두는 폴더입니다. `best.pt`는 GitHub Release asset에서 별도로 다운로드해야 합니다. |

## 설치

```bash
cd crackseg-repository
conda create -n crackseg python=3.10 -y
conda activate crackseg

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

CUDA 사용 가능 여부 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 모델 Weight

Repository 내 GitHub Release asset에서 `best.pt`를 다운로드한 뒤 `weights/` 폴더 아래에 배치해야 합니다.

다운로드한 모델 파일은 아래 위치로 옮깁니다.

```text
weights/best.pt
```

## 단일 이미지 손상 탐지

단일 이미지 추론은 `scripts/predict.py`를 사용합니다. 아래 예시는 `data/daechung_298.tif`를 입력으로 받아 `outputs/daechung_298_test`에 결과를 저장합니다.

```bash
cd crackseg-repository
conda activate crackseg

python scripts/predict.py \
  --image data/daechung_298.tif \
  --checkpoint weights/best.pt \
  --device cuda \
  --output-dir outputs/daechung_298_test
```

CPU 환경에서 실행하려면 `--device cpu`로 변경합니다. GPU가 있고 CUDA가 정상적으로 설정된 경우 `--device cuda`를 사용할 수 있습니다.

## 다중 이미지 손상 탐지 (Batch 단위)

폴더 단위 추론은 `scripts/batch_predict.py`를 사용합니다. 아래 예시는 `data/` 폴더 안의 지원 이미지 파일을 모두 처리하고 `outputs/predictions`에 결과를 저장합니다.

```bash
cd crackseg-repository
conda activate crackseg

python scripts/batch_predict.py \
  --input-dir data \
  --checkpoint weights/best.pt \
  --device cuda \
  --output-dir outputs/predictions
```

지원하는 데이터 형식:

```text
.jpg .jpeg .png .tif .tiff
```

## 입력 데이터 설명

입력 데이터는 RGB 이미지 또는 TIFF/GeoTIFF 형식이어야 합니다.

일반 이미지 파일인 `.jpg`, `.jpeg`, `.png`는 RGB 이미지로 읽어서 segmentation을 수행하며, `.tif`, `.tiff` 파일도 지원됩니다.

다중 이미지 손상 탐지는 지정한 입력 폴더 안의 지원 확장자 파일을 대상으로 수행됩니다. 입력 파일명은 출력 파일명 prefix로 사용되므로, 서로 다른 이미지가 같은 stem을 갖지 않도록 관리하는 것이 좋습니다.

## 출력 구조

입력 파일이 `daechung_298.tif`인 경우 `--output-dir outputs/daechung_298_test` 아래에 결과가 저장됩니다.

| 출력 구분 | 저장 위치 예시 | 내용 |
| --- | --- | --- |
| 전체 mask | `masks/daechung_298_mask.png` | 모든 손상 클래스를 ID 값으로 저장한 segmentation mask |
| 전체 overlay | `overlays/daechung_298_overlay.png` | 원본 이미지 위에 전체 손상 예측 결과를 색상으로 합성한 이미지 |
| confidence map | `confidence/daechung_298_confidence.png` | 모델 예측 confidence를 시각화한 이미지 |
| Pixel summary | `summaries/daechung_298_pixel_summary.csv` | 손상 유형별 pixel 수, 비율, 면적 등 정량 지표 |
| Metadata | `summaries/daechung_298_metadata.json` | 입력 파일, 이미지 크기, GeoTIFF 좌표 정보 등 메타데이터 |
| Instance detail | `instances/daechung_298_instances.csv` | 손상 instance별 중심점, bounding box, 면적 정보 |
| Instance summary | `instances/daechung_298_instance_summary.csv` | 손상 유형별 instance 개수와 요약 통계 |
| 손상 유형별 결과 | `by_damage/01_CRC/`, `by_damage/02_DLM/`, `by_damage/03_SPL/` | 균열, 박리, 박락을 손상 유형별로 분리 저장 |

손상 유형별 결과는 아래처럼 같은 구조로 저장됩니다.

| 손상 유형 | 폴더 | 저장되는 파일 예시 |
| --- | --- | --- |
| CRC - 균열 | `by_damage/01_CRC/` | `masks/daechung_298_CRC_binary_mask.png`, `overlays/daechung_298_CRC_overlay.png`, `summaries/daechung_298_CRC_pixel_summary.csv`, `instances/daechung_298_CRC_instances.csv` |
| DLM - 박리 | `by_damage/02_DLM/` | `masks/`, `overlays/`, `summaries/`, `instances/` 아래에 DLM 결과 저장 |
| SPL - 박락 | `by_damage/03_SPL/` | `masks/`, `overlays/`, `summaries/`, `instances/` 아래에 SPL 결과 저장 |

## 정량 출력

`summaries/<image>_pixel_summary.csv`에는 손상 클래스별 pixel 기반 요약 통계가 저장됩니다.

| 지표 | CSV 컬럼 | 의미 | 단위 / 조건 |
| --- | --- | --- | --- |
| 손상 유형 ID | `class_id` | 예측 클래스 ID | 0: BG, 1: CRC, 2: DLM, 3: SPL |
| 손상 유형 | `class_name` | 예측 클래스 이름 | BG, CRC, DLM, SPL |
| 손상 픽셀 수 | `pixel_count` | 해당 클래스로 예측된 pixel 개수 | pixel |
| 손상 픽셀 비율 | `pixel_ratio` | 전체 이미지 pixel 중 해당 클래스가 차지하는 비율 | 0-1 범위 |
| 손상율 | `area_percent` | `pixel_ratio`를 백분율로 표현한 값 | % |
| 손상 면적 | `area_m2` | GeoTIFF pixel scale이 있을 때 계산되는 실제 면적 | m², GeoTIFF scale 필요 |
| Pixel 크기 | `pixel_size_x_m`, `pixel_size_y_m` | GeoTIFF에서 읽은 x/y 방향 pixel 실제 크기 | meter, GeoTIFF scale 필요 |

`instances/<image>_instances.csv`에는 connected component 기반 instance 요약 정보가 저장됩니다.

| 지표 | CSV 컬럼 | 의미 | 단위 / 조건 |
| --- | --- | --- | --- |
| Instance ID | `instance_id` | 같은 클래스 내에서 분리된 예측 객체 번호 | 정수 |
| 손상 유형 ID | `class_id` | 예측 클래스 ID | 1: CRC, 2: DLM, 3: SPL |
| 손상 유형 | `class_name` | 예측 클래스 이름 | CRC, DLM, SPL |
| 중심점 by pixel | `centroid_x_px`, `centroid_y_px` | instance 중심점의 pixel 좌표 | pixel |
| 중심점 by meter | `centroid_x_m`, `centroid_y_m` | GeoTIFF 좌표 정보가 있을 때 계산되는 local meter 단위 중심점 좌표 | meter, GeoTIFF scale/tiepoint 필요 |
| BBox by pixel | `bbox_xmin_px`, `bbox_ymin_px`, `bbox_xmax_px`, `bbox_ymax_px` | instance를 감싸는 bounding box의 pixel 좌표 | pixel |
| BBox by meter | `bbox_xmin_m`, `bbox_ymin_m`, `bbox_xmax_m`, `bbox_ymax_m` | GeoTIFF 좌표 정보가 있을 때 계산되는 local meter 단위 bounding box 좌표 | meter, GeoTIFF scale/tiepoint 필요 |
| 손상 면적 by pixel | `area_px` | instance가 차지하는 pixel 면적 | pixel² |
| 손상 면적 by meter | `area_m2` | GeoTIFF pixel scale이 있을 때 계산되는 instance 실제 면적 | m², GeoTIFF scale 필요 |

## 추론 Tiling

기본 tile 설정은 `configs/dachung.yaml`에 정의되어 있습니다.

```yaml
img_size: 768
stride: 384
```

실행 시점에 아래처럼 tile 크기와 stride를 덮어쓸 수 있습니다.

```bash
python scripts/predict.py \
  --image data/daechung_298.tif \
  --checkpoint weights/best.pt \
  --tile-size 512 \
  --stride 256
```

## 테스트

```bash
cd crackseg-repository
python -m pytest -q
```
