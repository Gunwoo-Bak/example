# Concrete Crack 3D 작업 인수인계

## 목적

이 작업 폴더는 기존 RGB 손상 탐지 Git 코드에 DJI 드론 이미지와 3D mesh를 이용한 절대좌표 추출 기능을 추가한 최종 전달본입니다.

최종 목표는 다음과 같습니다.

- DJI RGB 이미지에서 균열, 박리, 박락 손상 탐지
- 기존 2D `instances.csv` 구조 유지
- 손상 instance별 대표 `X, Y, Z` 절대좌표 추가
- 손상 contour node 좌표 리스트를 2D와 3D로 함께 저장

## 남겨둔 필수 데이터

최종 파이프라인에 필요한 데이터만 남겼습니다.

| 경로 | 설명 |
| --- | --- |
| `ground_RGB/data/` | DJI 원본 이미지, `Timestamp.MRK`, PPK nav/obs/raw 파일 |
| `dam - Cloud.obj` | CloudCompare에서 생성한 댐 표면 mesh |
| `ground_RGB/` | 원본 Git의 RGB 코드와 3D 좌표 추출 수정 코드 |
| `ground_RGB/weights/best.pt` | 손상 탐지 모델 checkpoint |
| `ground_RGB/outputs/drone_predictions_mesh_3d/` | 현재 생성되어 있는 mesh-ray 기반 결과 폴더 |

GCP, DEM, LAS, 임시 PLY/BIN/TIF, projection check, LAS 실험 출력은 최종 경로에서 사용하지 않아 정리했습니다.

## 최종 실행 방식

최종 방식은 `mesh-ray`입니다.

원리는 다음과 같습니다.

```text
손상 pixel 또는 contour node
-> DJI XMP/MRK 기반 camera pose
-> camera ray 생성
-> dam - Cloud.obj mesh 표면과 교차
-> world X, Y, Z 저장
```

## 전체 배치 실행 명령

다음 작업자가 전체 7장 결과를 같은 형식으로 다시 만들 때 사용하면 됩니다.

```bash
cd /home/gunwoo/RGB_v2/ground_RGB
source /home/scsi/anaconda3/etc/profile.d/conda.sh
conda activate crack

python scripts/batch_predict.py \
  --input-dir data \
  --checkpoint weights/best.pt \
  --device cuda \
  --output-dir outputs/drone_predictions_mesh_3d \
  --mesh "/home/gunwoo/RGB_v2/dam - Cloud.obj" \
  --mrk data/DJI_202507021616_003_Timestamp.MRK \
  --geo3d-mode mesh-ray \
  --mesh-representative-mode centroid \
  --mesh-node-samples 256 \
  --mesh-ray-batch-size 128 \
  --mesh-ray-backend warp \
  --warp-device cuda:0
```

## 단일 이미지 실행 명령

특정 이미지 하나만 다시 만들 때 사용합니다.

```bash
cd /home/gunwoo/RGB_v2/ground_RGB
source /home/scsi/anaconda3/etc/profile.d/conda.sh
conda activate crack

python scripts/predict.py \
  --image data/DJI_20250702162019_0001_V.JPG \
  --checkpoint weights/best.pt \
  --device cuda \
  --output-dir outputs/drone_predictions_mesh_3d \
  --mesh "/home/gunwoo/RGB_v2/dam - Cloud.obj" \
  --mrk data/DJI_202507021616_003_Timestamp.MRK \
  --geo3d-mode mesh-ray \
  --mesh-representative-mode centroid \
  --mesh-node-samples 256 \
  --mesh-ray-batch-size 128 \
  --mesh-ray-backend warp \
  --warp-device cuda:0
```

## 핵심 출력 파일

```text
ground_RGB/outputs/drone_predictions_mesh_3d/instances/*_instances.csv
```

각 row는 손상 instance 1개를 의미합니다. 기존 2D 손상 정보에 아래 3D 컬럼이 추가됩니다.

| 컬럼 | 의미 |
| --- | --- |
| `world_x_m`, `world_y_m`, `world_z_m` | 손상 instance 대표 3D 좌표 |
| `mesh_ray_hit` | 대표 좌표가 mesh와 교차했는지 여부 |
| `instance_mesh_sample_count` | 대표 좌표 계산에 사용한 ray 수. centroid 모드에서는 보통 1 |
| `instance_mesh_hit_count` | sample 중 mesh 교차에 성공한 수 |
| `instance_mesh_hit_ratio` | sample hit 성공 비율 |
| `nodes_image_xy_json` | 손상 contour node의 2D image pixel 좌표 리스트 |
| `nodes_world_xyz_json` | 같은 contour node의 3D world 좌표 리스트 |
| `node_count` | 저장된 contour node 수 |
| `node_xyz_valid_count` | 3D 좌표 계산에 성공한 node 수 |
| `node_xyz_hit_ratio` | node hit 성공 비율 |

`nodes_image_xy_json` 예시는 다음과 같습니다.

```json
[[2934.5,922.0],[2935.0,922.5]]
```

`nodes_world_xyz_json` 예시는 다음과 같습니다.

```json
[[243040.19769633,431269.40379002,85.6082164],[243040.19750265,431269.40332637,85.60772349]]
```

## 옵션 설명

| 옵션 | 의미 |
| --- | --- |
| `--geo3d-mode mesh-ray` | mesh 표면과 camera ray를 교차시켜 3D 좌표를 계산 |
| `--mesh-representative-mode centroid` | 손상 instance 중심점 1개 ray로 대표 좌표 계산 |
| `--mesh-instance-samples 128` | `--mesh-representative-mode median`일 때만 사용하는 내부 pixel 최대 개수 |
| `--mesh-node-samples 256` | CSV에 저장할 손상 contour node 최대 개수 |
| `--mesh-ray-batch-size 128` | mesh ray 교차를 한 번에 처리할 chunk 크기 |
| `--mesh-ray-backend warp` | NVIDIA Warp 기반 GPU ray 교차 backend 사용 |
| `--warp-device cuda:0` | Warp가 사용할 CUDA device |

기본 대표 좌표는 centroid 1개 ray이므로 빠릅니다. `--mesh-node-samples`를 크게 하면 node 좌표가 촘촘해지지만 실행 시간이 늘어납니다.

## 병목 최적화 메모

가장 큰 병목은 GPU segmentation이 아니라 mesh ray 교차입니다. 기존 CPU `trimesh` backend에 더해 NVIDIA Warp 기반 `--mesh-ray-backend warp`를 추가했습니다. 현재 코드는 한 이미지의 모든 손상 instance sample ray와 contour node ray를 모아서 batch 교차하도록 개선되어 있습니다.

실제 병목을 확인하려면 `--profile`을 추가합니다. 이 옵션은 mesh ray chunk별 tqdm progress bar와 단계별 소요 시간을 출력합니다.

```bash
python scripts/predict.py \
  --image data/DJI_20250702162031_0013_V.JPG \
  --checkpoint weights/best.pt \
  --device cuda \
  --output-dir outputs/profile_mesh_3d \
  --mesh "/home/gunwoo/RGB_v2/dam - Cloud.obj" \
  --mrk data/DJI_202507021616_003_Timestamp.MRK \
  --geo3d-mode mesh-ray \
  --mesh-representative-mode centroid \
  --mesh-node-samples 64 \
  --mesh-ray-batch-size 128 \
  --mesh-ray-backend warp \
  --warp-device cuda:0 \
  --profile
```

특히 아래 항목을 보면 병목 위치를 판단할 수 있습니다.

| Profile 항목 | 의미 |
| --- | --- |
| `script_inference_sec` | GPU segmentation 추론 시간 |
| `export_instances_3d_sec` | 3D 좌표 계산 전체 시간 |
| `mesh_instance_intersection_sec` | 대표 좌표용 mesh ray 교차 시간 |
| `mesh_node_intersection_sec` | contour node용 mesh ray 교차 시간 |
| `mesh_total_ray_count` | 계산한 전체 ray 개수 |

속도가 더 필요하면 우선 샘플 수를 낮춥니다.

```bash
--mesh-representative-mode centroid \
--mesh-node-samples 64 \
--mesh-ray-batch-size 128 \
--mesh-ray-backend warp
```

`Killed`가 뜨면 ray 교차 중 메모리 피크가 난 것입니다. `--mesh-ray-batch-size 32` 또는 `16`으로 더 낮춥니다. CUDA/Warp 환경 문제가 있으면 `--mesh-ray-backend trimesh`로 기존 CPU backend를 사용할 수 있습니다.

정밀 검사용으로 대표 좌표도 내부 pixel median으로 계산하고 싶다면 `--mesh-representative-mode median --mesh-instance-samples 128`을 사용합니다.

## 현재 상태

- 코드 수정은 `ground_RGB`에 반영되어 있습니다.
- `DJI_20250702162019_0001_V_instances.csv`는 node JSON 컬럼이 들어가는 것을 확인했습니다.
- 전체 파일 재처리는 다음 작업자가 위 batch 명령으로 실행하면 됩니다.
- GPU가 있으면 `--device cuda`를 사용합니다.
