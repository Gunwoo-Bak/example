# Sensor Damage Detection Repository

이 저장소는 `상위 센서 -> 하위 센서 -> 탐지/손상 유형 -> 알고리즘 소스코드` 구조를 그대로 GitHub에서 보이도록 만든 템플릿이다.

## 폴더 구조 원칙
- GitHub는 **빈 폴더를 표시하지 않으므로**, 각 마지막 폴더(탐지/손상 유형) 안에 예시 코드 파일을 넣어 두었다.
- 하위 센서가 따로 없는 센서(GPR, TIR, STIV, MBES)는 계층 통일을 위해 `general/` 폴더를 두었다.
- 마지막 폴더에는 `train.py`, `infer.py`, `model.py`, `config.yaml`, `README.md`가 들어 있다.

## Repository Tree
```text
sensor_damage_detection_repo/
├─ rgb/
│  ├─ ground_rgb/
│  │  ├─ delamination/
│  │  ├─ spalling/
│  │  ├─ corrosion_rebar_exposure/
│  │  └─ crack/
│  └─ underwater_rgb/
│     ├─ delamination/
│     ├─ spalling/
│     ├─ corrosion_rebar_exposure/
│     └─ crack/
├─ lidar/
│  ├─ ground_lidar/
│  │  ├─ delamination/
│  │  ├─ spalling/
│  │  ├─ crack/
│  │  └─ deformation_displacement/
│  └─ underwater_lidar/
│     └─ scour_and_erosion/
├─ gpr/
│  └─ general/
│     ├─ delamination/
│     ├─ spalling/
│     ├─ corrosion_rebar_exposure/
│     ├─ crack/
│     └─ leakage_efflorescence/
├─ tir/
│  └─ general/
│     ├─ delamination/
│     ├─ spalling/
│     ├─ leakage_efflorescence/
│     ├─ crack/
│     └─ corrosion_rebar_exposure/
├─ stiv/
│  └─ general/
│     └─ flow_velocity_and_discharge/
└─ mbes/
   └─ general/
      └─ scour_and_erosion/
```

## Naming Guide
- `rgb`: RGB 영상 기반 센서
- `lidar`: LiDAR 기반 센서
- `gpr`: Ground Penetrating Radar
- `tir`: Thermal Infrared
- `stiv`: Space-Time Image Velocimetry
- `mbes`: Multibeam Echo Sounder

필요하면 이 구조를 그대로 새 GitHub 저장소에 업로드한 뒤, 실제 알고리즘 코드로 교체하면 된다.
