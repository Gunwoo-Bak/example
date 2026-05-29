# CrackSeg

콘크리트 표면 손상 이미지에 대한 semantic segmentation 추론용 저장소입니다. RGB 이미지 또는 TIFF/GeoTIFF 파일을 입력으로 받아 아래 세 가지 손상 클래스를 예측합니다.

| ID | Class | 의미 | Overlay 색상 |
| ---: | --- | --- | --- |
| 0 | BG | 배경 | black |
| 1 | CRC | crack / 균열 | red |
| 2 | DLM | delamination / 박리 | green |
| 3 | SPL | spalling / 박락 | orange |

이 저장소는 추론 실행에 초점을 둔 코드 저장소입니다. 학습 데이터, GT mask, 모델 weight, 생성된 output 파일은 포함하지 않습니다.
추론을 실행하기 전에 GitHub Release asset에서 모델 checkpoint를 별도로 다운로드해야 합니다.

## 주요 기능

- 단일 이미지 및 폴더 단위 batch segmentation 지원
- RGB 이미지와 TIFF/GeoTIFF 입력 지원
- mask, overlay, confidence map, pixel summary, instance summary 출력
- `CRC`, `DLM`, `SPL` 손상 유형별 결과 폴더 분리 저장
- GeoTIFF의 scale/tiepoint 정보를 이용한 local meter 좌표 및 `m2` 면적 요약 지원
- 본 저장소의 정량 결과는 예측 결과 기반 통계입니다. Accuracy, IoU, F1, precision, recall 같은 GT 기반 성능 평가는 별도의 GT mask가 필요합니다.

## 저장소 구조

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

Git에서 제외되는 로컬 폴더 및 파일:

```text
data/       입력 이미지, commit 제외
outputs/    생성 결과, commit 제외
weights/*.pt
```

## 설치

```bash
cd /home/gunwoo/CrackSegmentation/crackseg-repository
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

모델 checkpoint는 파일 크기가 크기 때문에 Git에 포함하지 않습니다. 저장소의 GitHub Release asset에서 `best.pt`를 다운로드한 뒤 `weights/` 폴더 아래에 배치해야 합니다.

예상되는 로컬 구조:

```text
crackseg-repository/
  weights/
    best.pt
```

Release page에서 직접 다운로드 URL을 제공하는 경우 아래처럼 받을 수 있습니다.

```bash
cd /home/gunwoo/CrackSegmentation/crackseg-repository
mkdir -p weights
wget -O weights/best.pt "https://github.com/Gunwoo-Bak/example/releases/download/v1.0.0/best.pt"
```

또는 아래 Release 페이지에서 `best.pt`를 직접 다운로드합니다.

```text
https://github.com/<OWNER>/<REPOSITORY>/releases
```

다운로드한 파일은 아래 위치로 옮깁니다.

```text
weights/best.pt
```

`--checkpoint` 인자에 절대 경로를 직접 넘겨도 됩니다.

현재 작업 서버의 로컬 checkpoint 예시:

```text
/home/gunwoo/CrackSegmentation/outputs/dachung_finetune/best.pt
```

Release를 생성하는 관리자는 아래 파일을 binary asset으로 첨부하면 됩니다.

```text
/home/gunwoo/CrackSegmentation/outputs/dachung_finetune/best.pt
```

## 단일 이미지 추론

```bash
cd /home/gunwoo/CrackSegmentation/crackseg-repository
conda activate crackseg

python scripts/predict.py \
  --image data/daechung_298.tif \
  --checkpoint weights/best.pt \
  --device cuda \
  --output-dir outputs/daechung_298_test
```

## Batch 추론

```bash
cd /home/gunwoo/CrackSegmentation/crackseg-repository
conda activate crackseg

python scripts/batch_predict.py \
  --input-dir data \
  --checkpoint weights/best.pt \
  --device cuda \
  --output-dir outputs/predictions
```

지원하는 입력 확장자:

```text
.jpg .jpeg .png .tif .tiff
```

## 입력 데이터 설명

입력 데이터는 RGB 이미지 또는 TIFF/GeoTIFF 형식이어야 합니다. 단일 이미지 추론에서는 `--image`에 하나의 파일 경로를 입력하고, batch 추론에서는 `--input-dir`에 이미지들이 들어 있는 폴더를 입력합니다.

일반 이미지 파일인 `.jpg`, `.jpeg`, `.png`는 RGB 이미지로 읽어서 segmentation을 수행합니다. `.tif`, `.tiff` 파일도 지원하며, GeoTIFF에 `ModelPixelScaleTag`와 `ModelTiepointTag`가 포함되어 있으면 출력 CSV에 local meter 단위 좌표와 면적 정보가 추가됩니다.

Batch 추론은 지정한 입력 폴더 안의 지원 확장자 파일을 대상으로 수행됩니다. 입력 파일명은 출력 파일명 prefix로 사용되므로, 서로 다른 이미지가 같은 stem을 갖지 않도록 관리하는 것이 좋습니다.

## 출력 구조

입력 파일이 `daechung_298.tif`인 경우 결과는 아래와 같은 구조로 저장됩니다.

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

손상 유형별 출력 예시:

```text
by_damage/01_CRC/masks/daechung_298_CRC_binary_mask.png
by_damage/01_CRC/overlays/daechung_298_CRC_overlay.png
by_damage/01_CRC/summaries/daechung_298_CRC_pixel_summary.csv
by_damage/01_CRC/instances/daechung_298_CRC_instances.csv
```

## 정량 출력

`summaries/<image>_pixel_summary.csv`에는 클래스별 pixel 기반 요약 통계가 저장됩니다.

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

주요 컬럼의 의미는 다음과 같습니다.

- `pixel_count`: 해당 클래스로 예측된 pixel 개수입니다.
- `pixel_ratio`: 전체 이미지 pixel 중 해당 클래스가 차지하는 비율입니다.
- `area_percent`: `pixel_ratio`를 백분율로 표현한 값입니다.
- `area_m2`: GeoTIFF pixel scale이 있을 때 계산되는 실제 면적입니다. 단위는 제곱미터입니다.
- `pixel_size_x_m`, `pixel_size_y_m`: GeoTIFF에서 읽은 pixel의 x/y 방향 실제 크기입니다. 단위는 meter입니다.

`instances/<image>_instances.csv`에는 connected component 기반 instance 요약 정보가 저장됩니다.

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

주요 컬럼의 의미는 다음과 같습니다.

- `instance_id`: 같은 클래스 내에서 분리된 예측 객체 번호입니다.
- `class_id`, `class_name`: 예측된 손상 클래스 ID와 이름입니다.
- `centroid_x_px`, `centroid_y_px`: instance 중심점의 pixel 좌표입니다.
- `bbox_xmin_px`, `bbox_ymin_px`, `bbox_xmax_px`, `bbox_ymax_px`: instance를 감싸는 bounding box의 pixel 좌표입니다.
- `area_px`: instance가 차지하는 pixel 면적입니다.
- `centroid_x_m`, `centroid_y_m`: GeoTIFF 좌표 정보가 있을 때 계산되는 local meter 단위 중심점 좌표입니다.
- `bbox_xmin_m`, `bbox_ymin_m`, `bbox_xmax_m`, `bbox_ymax_m`: GeoTIFF 좌표 정보가 있을 때 계산되는 local meter 단위 bounding box 좌표입니다.
- `area_m2`: GeoTIFF pixel scale이 있을 때 계산되는 instance 실제 면적입니다. 단위는 제곱미터입니다.

주의할 점은 이 값들이 **모델 예측 결과를 요약한 통계**라는 것입니다. GT mask와 비교해서 계산하는 accuracy, IoU, F1, precision, recall 같은 성능 지표가 아닙니다. 이런 성능 평가는 GT mask가 별도로 있어야 계산할 수 있습니다.

## GeoTIFF 참고 사항

TIFF 파일에 `ModelPixelScaleTag`와 `ModelTiepointTag`가 포함되어 있으면 CSV 출력에 local meter 단위 좌표 필드가 추가됩니다.

CRS 텍스트가 `Arbitrary (m)`인 경우, 출력 좌표는 위도/경도 또는 EPSG 기반 좌표가 아니라 local meter 좌표입니다.

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
cd /home/gunwoo/CrackSegmentation/crackseg-repository
python -m pytest -q
```
