# LiDAR OpenGL Viewer

순수 CMake + OpenGL로 `.bin` 포인트클라우드를 띄우는 최소 뷰어입니다.

## Build

```bash
cd /home/juno/lidar
cmake -S . -B build
cmake --build build -j
```

## Structure

- `src/main.cpp`: 최소 엔트리포인트
- `src/viewer_app.cpp`: OpenGL 렌더링, 입력, HUD, 패널 UI
- `src/data_io.cpp`: KITTI / Ouster `.bin` 로딩
- `include/lidar_viewer/data_io.hpp`: 로더가 공유하는 데이터 구조
- `src/algorithms/`: 노면 제거, 클러스터링, detection placeholder 구현
- `include/lidar_viewer/app.hpp`: 앱 진입 선언
- `include/lidar_viewer/algorithms/`: 알고리즘 인터페이스와 공통 타입

## Run

인자를 생략하면 기본으로 gangnam Ouster 데이터를 엽니다.

```bash
./build/lidar_viewer
```

```bash
./build/lidar_viewer
```

## Controls

- `Left drag`: orbit
- `Right drag`: pan
- `Wheel`: zoom
- `N` or `Right`: next frame
- `P` or `Left`: previous frame
- `F`: fit view
- `C`: intensity / height color toggle
- `[` `]`: point size
- `SAVE` button in the grid panel: persist current viewer settings
- `GROUND` button: toggle RANSAC ground removal
- `CLUST` button: toggle OGM/DFS clustering boxes
- calibration panel: drag `yaw / pitch / roll / z` sliders for manual alignment
- calibration value boxes: click a value, type a number, press `Enter`
- `Esc`: quit

## Notes

- KITTI 스타일 `float32 x, y, z, intensity` `.bin` 파일을 바로 지원합니다.
- Ouster `OS1-64 LEGACY` 레코드 컨테이너도 지원합니다.
- dataset profile은 입력 경로로 자동 감지됩니다. `gangnam`, `kitti`, `default` 프로파일을 지원합니다.
- 저장된 설정은 `config/profiles/<profile>.cfg`에 기록되고 다음 실행 때 자동으로 불러옵니다.
- `GROUND` 토글 상태도 설정 파일에 함께 저장됩니다.
- calibration rotation과 z translation 값도 같은 설정 파일에 함께 저장됩니다.
- 데이터 누락 없이 기본값으로 모든 포인트를 그대로 GPU에 올려서 그립니다.
- 노면 제거는 `lidar_viewer::algorithms::RunGroundRemovalRansac()`에 RANSAC plane 방식으로 들어가 있습니다.
- 클러스터링은 OGM grid + DFS connected component 방식으로 들어가 있으며, bbox를 반투명 박스로 표시합니다.
- detection은 아직 placeholder이며, `lidar_viewer::algorithms::RunPipelinePlaceholder()`를 기준으로 실제 구현을 이어 붙일 수 있습니다.
# lidar
# lidar
