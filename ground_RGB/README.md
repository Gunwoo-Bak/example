# RGB Drone Image 손상 탐지

콘크리트 표면 RGB 이미지에 대한 손상 segmentation 모델 파이프라인입니다. RGB 이미지 또는 TIFF/GeoTIFF 파일을 입력으로 받아 아래 세 가지 손상 클래스를 예측합니다.

| ID | Class | 의미 | Overlay 색상 |
| ---: | --- | --- | --- |
| 0 | BG | 배경 | black |
| 1 | CRC | crack / 균열 | red |
| 2 | DLM | delamination / 박리 | green |
| 3 | SPL | spalling / 박락 | orange |

모델을 실행하기 전에 GitHub Release asset에서 모델 checkpoint(best.pt)와 3D mesh 모델(dam - Cloud.obj) 파일을 별도로 다운로드해야 합니다.
모델 checkpoint는 `weights/best.pt` 위치에 있어야 합니다. 

아래 실행 명령어에서 `RGB_V2_DIR`는 사용자가 이 폴더를 내려받아 둔 상위 경로를 의미합니다. 예를 들어 `RGB_v2` 폴더를 `/workspace/RGB_v2`에 두었다면 `RGB_V2_DIR="/workspace/RGB_v2"`처럼 지정해 실행합니다.

## 주요 기능

- 단일 이미지 및 폴더 단위 batch segmentation 지원
- RGB 이미지와 TIFF/GeoTIFF 입력 지원
- mask, overlay, confidence map, pixel summary, instance summary 출력
- 균열, 박리, 박락 손상 유형별 결과 폴더 분리 저장
- GeoTIFF의 scale/tiepoint 정보를 이용한 local meter 좌표 및 산출 면적 지원
- DJI 원본 이미지 + MRK + mesh OBJ를 이용한 손상 instance 3D 좌표 및 contour node 3D 좌표 저장
- 모델 예측 결과 기반 정량 결과 산출

## 폴더 구조

```text
RGB_v2/
  dam - Cloud.obj
  HANDOFF.md
  ground_RGB/
    configs/
      dachung.yaml
    data/
      [input images(.jpg, .jpeg, .png)]
    scripts/
      predict.py
      batch_predict.py
    src/crackseg/
      config.py
      export.py
      geotiff.py
      geo3d.py
      inference.py
      instances.py
      io.py
      las_ray.py
      mesh_ray.py
      metrics.py
      models.py
      visualization.py
    tests/
      test_metrics.py
      test_geo3d.py
      test_las_ray.py
      test_mesh_ray.py
    weights/
      README.md
      best.pt
    outputs/
```

주요 파일과 폴더의 역할은 다음과 같습니다.

| 경로 | 설명 |
| --- | --- |
| `configs/dachung.yaml` | 탐지에 사용하는 기본 설정 파일입니다. 클래스 이름, 모델 구조, tile 크기, stride, overlay 색상 팔레트가 정의되어 있습니다. |
| `scripts/predict.py` | 단일 이미지 segmentation을 수행하는 실행 스크립트입니다. |
| `scripts/batch_predict.py` | 입력 폴더 안의 여러 이미지를 한 번에 처리하는 batch 실행 스크립트입니다. |
| `data/` | 실행 입력 데이터 폴더입니다. DJI 원본 JPG, `Timestamp.MRK`, PPK 파일과 예제 이미지가 들어 있습니다. |
| `src/crackseg/config.py` | YAML 설정을 읽고 탐지 설정 객체로 변환합니다. |
| `src/crackseg/models.py` | segmentation 모델 구조를 생성합니다. |
| `src/crackseg/inference.py` | checkpoint 로드와 sliding-window 기반 추론을 수행합니다. |
| `src/crackseg/io.py` | RGB 이미지, TIFF/GeoTIFF 입력 파일을 읽고 batch 대상 이미지 목록을 구성합니다. |
| `src/crackseg/export.py` | mask, overlay, confidence map, summary CSV, metadata JSON 등 결과 파일을 저장합니다. |
| `src/crackseg/geotiff.py` | GeoTIFF scale/tiepoint 정보를 읽어 pixel 좌표를 local meter 좌표로 변환합니다. |
| `src/crackseg/geo3d.py` | 3D 좌표 enrich 공통 진입점과 선택적 orthophoto/GCP 경로를 제공합니다. |
| `src/crackseg/las_ray.py` | DJI XMP/MRK camera pose 해석과 LAS 기반 보조 3D 경로를 제공합니다. |
| `src/crackseg/mesh_ray.py` | DJI camera ray와 triangulated mesh를 교차시켜 손상 instance 및 contour node의 3D 좌표를 계산합니다. |
| `src/crackseg/metrics.py` | 클래스별 pixel 수, pixel 비율, 손상율, 면적 등 정량 지표를 계산합니다. |
| `src/crackseg/instances.py` | connected component 기반으로 손상 instance를 분리하고 중심점, bounding box, 면적을 계산합니다. |
| `src/crackseg/visualization.py` | 예측 mask를 색상 overlay와 confidence map으로 시각화합니다. |
| `tests/test_metrics.py` | 정량 지표 계산 로직에 대한 테스트 코드입니다. |
| `weights/` | 모델 checkpoint를 두는 폴더입니다. 실행 시 `weights/best.pt`를 사용합니다. |
| `$RGB_V2_DIR/dam - Cloud.obj` | 3D 좌표 산출에 사용하는 dam surface mesh입니다. |

## 설치

```bash
RGB_V2_DIR="/path/to/RGB_v2"
cd "$RGB_V2_DIR/ground_RGB"
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

모델 checkpoint는 아래 위치에 있어야 합니다.

```text
weights/best.pt
```

현재 전달 폴더에는 `weights/best.pt`가 포함되어 있습니다. 파일이 없거나 교체 모델을 사용하는 경우 동일한 경로에 새 checkpoint를 배치합니다.

## 단일 이미지 손상 탐지

단일 이미지 추론은 `scripts/predict.py`를 사용합니다. 아래 예시는 `data/daechung_298.tif`를 입력으로 받아 `outputs/daechung_298_test`에 결과를 저장합니다.

```bash
RGB_V2_DIR="/path/to/RGB_v2"
cd "$RGB_V2_DIR/ground_RGB"
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
RGB_V2_DIR="/path/to/RGB_v2"
cd "$RGB_V2_DIR/ground_RGB"
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

## DJI/Mesh 기반 3D 좌표 추출

최종 3D 경로는 DJI 원본 이미지의 XMP/RTK 메타데이터, `Timestamp.MRK`, 그리고 CloudCompare 등에서 만든 triangulated mesh OBJ를 사용합니다. 손상 탐지 결과의 2D pixel 좌표에서 camera ray를 쏘고, 그 ray가 mesh 표면과 만나는 지점을 world `X, Y, Z` 좌표로 저장합니다.

```text
DJI image pixel
-> DJI calibrated camera + MRK camera position
-> camera ray
-> mesh triangle intersection
-> representative damage XYZ + contour node XYZ 저장
```

현재 프로젝트의 최종 입력은 다음 세 가지입니다.

| 입력 | 경로 | 설명 |
| --- | --- | --- |
| DJI 이미지/항법 폴더 | `data` | JPG, MRK, PPK nav/obs/raw 포함 |
| Mesh OBJ | `$RGB_V2_DIR/dam - Cloud.obj` | CloudCompare에서 생성한 dam surface mesh |
| Model checkpoint | `weights/best.pt` | 손상 segmentation model |

다중이미지(Batch) 실행 예시는 다음과 같습니다. 작업자는 이 명령으로 최종 결과를 생성하면 됩니다.
"RGB_V2_DIR"를 작업 상위 경로로 수정하여 사용하면 됩니다.

```bash
RGB_V2_DIR="/path/to/RGB_v2"
cd "$RGB_V2_DIR/ground_RGB"
conda activate crackseg

python scripts/batch_predict.py \
  --input-dir data \
  --checkpoint weights/best.pt \
  --device cuda \
  --output-dir outputs/drone_predictions_mesh_3d \
  --mesh "$RGB_V2_DIR/dam - Cloud.obj" \
  --mrk data/DJI_202507021616_003_Timestamp.MRK \
  --geo3d-mode mesh-ray \
  --mesh-representative-mode centroid \
  --mesh-node-samples 256 \
  --mesh-ray-batch-size 128 \
  --mesh-ray-backend warp \
  --warp-device cuda:0
```

단일 이미지도 같은 옵션을 사용할 수 있습니다.

```bash
RGB_V2_DIR="/path/to/RGB_v2"
cd "$RGB_V2_DIR/ground_RGB"
conda activate crackseg

python scripts/predict.py \
  --image data/DJI_20250702162019_0001_V.JPG \
  --checkpoint weights/best.pt \
  --device cuda \
  --output-dir outputs/drone_predictions_mesh_3d \
  --mesh "$RGB_V2_DIR/dam - Cloud.obj" \
  --mrk data/DJI_202507021616_003_Timestamp.MRK \
  --geo3d-mode mesh-ray \
  --mesh-representative-mode centroid \
  --mesh-node-samples 256 \
  --mesh-ray-batch-size 128 \
  --mesh-ray-backend warp \
  --warp-device cuda:0
```

추가되는 주요 컬럼은 다음과 같습니다.

| CSV 컬럼 | 의미 |
| --- | --- |
| `world_x_m`, `world_y_m`, `world_z_m` | 손상 instance 대표 3D 좌표. 기본값은 2D instance 중심점 `centroid_x_px`, `centroid_y_px`에서 쏜 ray의 mesh hit입니다. |
| `geo3d_source` | 기본값은 `dji_pose_mesh_ray_centroid` |
| `xyz_valid` | 대표 3D 좌표가 계산되었는지 여부 |
| `mesh_ray_hit` | camera ray가 mesh 표면과 교차했는지 여부 |
| `mesh_ray_t_m` | camera center에서 mesh 교차점까지 ray 거리 |
| `mesh_ray_backend` | 사용한 ray 교차 backend. `warp`는 GPU/CPU Warp backend, `trimesh`는 기존 CPU backend입니다. |
| `instance_mesh_sample_count` | 대표 좌표 계산에 사용한 ray 수. centroid 모드에서는 보통 1입니다. |
| `instance_mesh_hit_count` | sample 중 mesh hit에 성공한 수 |
| `instance_mesh_hit_ratio` | sample hit 성공 비율 |
| `nodes_image_xy_json` | 손상 contour node의 2D image pixel 좌표 리스트. 예: `[[x,y], ...]` |
| `nodes_world_xyz_json` | 같은 contour node의 3D world 좌표 리스트. 예: `[[x,y,z], ...]` |
| `node_count` | 저장된 contour node 수 |
| `node_xyz_valid_count` | 3D 좌표 계산에 성공한 node 수 |
| `node_xyz_hit_ratio` | node mesh hit 성공 비율 |

`instances/<image>_instances.csv`의 기본 row 구조는 기존과 같습니다. 즉 손상 instance 1개가 CSV 1줄이며, 그 줄 안에 대표 좌표와 node 좌표 리스트가 추가됩니다. 기본 `--mesh-representative-mode centroid`에서는 손상 중심점 1개만 ray 교차해서 대표 좌표를 계산합니다. 예전처럼 손상 내부 pixel 여러 개의 median으로 대표 좌표를 만들고 싶을 때만 `--mesh-representative-mode median`과 `--mesh-instance-samples`를 사용합니다. `--mesh-node-samples`는 저장할 contour node의 최대 개수를 정합니다.

### Mesh-Ray 병목 최적화

3D 좌표 계산에서 가장 느린 부분은 mesh ray 교차입니다. 현재 코드는 instance 대표 좌표는 중심점 1개 ray로 계산하고, contour node ray만 `--mesh-node-samples` 개수만큼 계산합니다. 한 이미지 안의 ray를 모은 뒤 `--mesh-ray-batch-size` 단위로 batch 교차합니다.

GPU ray 교차를 쓰려면 NVIDIA Warp backend를 선택합니다. `warp` backend는 mesh와 camera origin을 로컬 좌표로 평행이동한 뒤 float32 GPU raycast를 수행하고, 결과를 다시 절대좌표로 복원합니다.

실제 병목을 확인하려면 `--profile`을 추가합니다. 이 옵션을 켜면 mesh ray chunk별 tqdm progress bar와 단계별 소요 시간이 출력됩니다.

```bash
RGB_V2_DIR="/path/to/RGB_v2"
cd "$RGB_V2_DIR/ground_RGB"
conda activate crackseg

python scripts/predict.py \
  --image data/DJI_20250702162031_0013_V.JPG \
  --checkpoint weights/best.pt \
  --device cuda \
  --output-dir outputs/profile_mesh_3d \
  --mesh "$RGB_V2_DIR/dam - Cloud.obj" \
  --mrk data/DJI_202507021616_003_Timestamp.MRK \
  --geo3d-mode mesh-ray \
  --mesh-representative-mode centroid \
  --mesh-node-samples 64 \
  --mesh-ray-batch-size 128 \
  --mesh-ray-backend warp \
  --warp-device cuda:0 \
  --profile
```

주요 profile 항목은 다음과 같습니다.

| Profile 항목 | 의미 |
| --- | --- |
| `script_load_model_sec` | model checkpoint 로딩 시간 |
| `script_inference_sec` | GPU segmentation 추론 시간 |
| `script_build_mesh_context_sec` | DJI pose 해석 및 mesh context 준비 시간 |
| `export_instances_2d_sec` | 2D connected component instance 계산 시간 |
| `export_instances_3d_sec` | 3D 좌표 enrich 전체 시간 |
| `mesh_prepare_regions_sec` | 손상 mask에서 instance sample/node 추출 시간 |
| `mesh_ray_direction_sec` | pixel 좌표를 camera/world ray로 변환하는 시간 |
| `mesh_instance_intersection_sec` | 대표 좌표용 mesh ray 교차 시간 |
| `mesh_node_intersection_sec` | contour node용 mesh ray 교차 시간 |
| `mesh_total_ray_count` | 이번 이미지에서 계산한 전체 ray 개수 |
| `script_save_outputs_sec` | 결과 파일 저장 전체 시간 |

`--mesh-ray-backend`는 다음 중 하나를 선택합니다.

| Backend | 의미 |
| --- | --- |
| `trimesh` | 기존 CPU ray 교차. 가장 보수적인 기본값입니다. |
| `warp` | NVIDIA Warp 기반 ray 교차. CUDA 환경에서는 GPU BVH를 사용합니다. |
| `auto` | Warp 초기화를 시도하고 실패하면 CPU `trimesh`로 fallback합니다. |

실행 시간이 너무 길면 먼저 아래 값을 낮춰보는 것이 좋습니다.

```bash
--mesh-representative-mode centroid \
--mesh-node-samples 64 \
--mesh-ray-batch-size 128 \
--mesh-ray-backend warp
```

`Killed`가 출력되면 OS가 메모리 피크 때문에 프로세스를 종료한 것입니다. 이 경우 ray chunk를 더 작게 낮춰 다시 실행합니다. CUDA/Warp 환경 문제가 있으면 `--mesh-ray-backend trimesh`로 기존 CPU backend를 사용할 수 있습니다.

```bash
--mesh-ray-batch-size 32
```

정밀 검사용으로 node를 더 촘촘히 저장하려면 다음처럼 키울 수 있지만 실행 시간도 증가합니다.

```bash
--mesh-instance-samples 128 \
--mesh-node-samples 256 \
--mesh-representative-mode median
```

## Orthophoto/GCP 기반 3D 좌표 추출

GeoTIFF orthophoto와 GCP CSV를 함께 제공하는 homography 기반 경로도 남아 있습니다. 다만 3D 구조물 균열 목적에서는 DJI/LAS 기반 경로를 우선 사용합니다.

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
| 대표 3D 좌표 | `world_x_m`, `world_y_m`, `world_z_m` | DJI/MRK pose와 mesh ray 교차로 계산한 손상 instance 대표 좌표 | meter |
| 2D node 리스트 | `nodes_image_xy_json` | 손상 contour node의 image pixel 좌표 JSON 리스트 | pixel |
| 3D node 리스트 | `nodes_world_xyz_json` | `nodes_image_xy_json`과 같은 순서의 3D world 좌표 JSON 리스트 | meter |
| Node 품질 | `node_count`, `node_xyz_valid_count`, `node_xyz_hit_ratio` | node 좌표 계산 성공 여부 확인용 지표 | count, ratio |

## 추론 Tiling

기본 tile 설정은 `configs/dachung.yaml`에 정의되어 있습니다.

```yaml
img_size: 768
stride: 384
```

실행 시점에 아래처럼 tile 크기와 stride를 덮어쓸 수 있습니다.

```bash
RGB_V2_DIR="/path/to/RGB_v2"
cd "$RGB_V2_DIR/ground_RGB"
conda activate crackseg

python scripts/predict.py \
  --image data/daechung_298.tif \
  --checkpoint weights/best.pt \
  --tile-size 512 \
  --stride 256
```

## 테스트

```bash
RGB_V2_DIR="/path/to/RGB_v2"
cd "$RGB_V2_DIR/ground_RGB"
python -m pytest -q
```
