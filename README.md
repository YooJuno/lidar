# LiDAR OpenGL Viewer

CMake + OpenGL 기반 LiDAR 뷰어입니다. `.bin`과 ROS 2 PointCloud2 bag을 지원합니다.

## Build

```bash
cd ~/workspace/lidar
source /opt/ros/jazzy/setup.bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
```

ROS 환경을 읽고 빌드하면 bag 입력이 활성화됩니다. ROS 없는 `.bin` 전용 빌드는
`-DLIDAR_VIEWER_ENABLE_ROSBAG=OFF`로 구성할 수 있습니다.
시스템 파일 선택 창에는 `zenity`가 필요합니다(Ubuntu: `sudo apt install zenity`). 파일 경로를 명령행에 직접 지정하는 실행과 `--inspect`는 파일 선택 창을 사용하지 않습니다.

UI는 고정 버전의 [Dear ImGui](https://github.com/ocornut/imgui/tree/f5befd2d29e66809cd1110a152e375a7f1981f06)와 GLFW/OpenGL 백엔드를 사용합니다. 최초 CMake 구성 시 소스를 받아 `build/_deps`에 보관하며, 이후에는 캐시를 사용합니다.

## Structure

- `src/main.cpp`: 최소 엔트리포인트
- `src/viewer_app.cpp`: OpenGL 렌더링, 입력, ImGui 인터페이스
- `include/lidar_viewer/ui_layout.hpp`: 창 크기·DPI별 패널 배치
- `include/lidar_viewer/settings_io.hpp`: 설정값 검증·원자적 저장
- `src/bag_io.cpp`: ROS 2 bag 직접 읽기·프레임 탐색
- `src/imu_alignment.cpp`: IMU 중력 추정·시각 대응·센서 축 변환
- `src/file_dialog.cpp`: 시스템 파일 선택 창과 비동기 결과 수신
- `include/lidar_viewer/camera_2d.hpp`: 회전한 2D 시점의 카메라·이동 좌표계
- `src/data_io.cpp`: KITTI / Ouster `.bin` 로딩
- `include/lidar_viewer/data_io.hpp`: 로더가 공유하는 데이터 구조
- `src/algorithms/`: 노면 제거, 클러스터링, detection placeholder 구현
- `include/lidar_viewer/app.hpp`: 앱 진입 선언
- `include/lidar_viewer/algorithms/`: 알고리즘 인터페이스와 공통 타입

## Run

입력 경로를 생략하면 **빈 뷰어가 먼저 열립니다.** 상단 **Open file** 버튼을 누르면 시스템 파일 탐색 창이 열립니다. bag 폴더 안의 `metadata.yaml` 또는 `.db3`·`.mcap` 파일을 선택하세요. 분할 파일을 골라도 같은 폴더의 기록 전체를 엽니다. 탐색 창을 취소해도 뷰어는 계속 열려 있습니다. bag을 열기 전에는 재생·프레임 이동·타임라인이 비활성화됩니다.

```bash
source /opt/ros/jazzy/setup.bash
./build/lidar_viewer --sensor-height 0.57 --imu-level --2d
```

BIN 파일이나 단일 bag 폴더를 뷰어에 드래그해 열 수도 있습니다. 하단은 소스 선택·폴더 등록·재생·타임라인에 사용하고, 캔버스 상단에서 2D/3D·전체 맞춤·ROI 도구를 바로 선택합니다.

입력 경로를 직접 주면 해당 데이터를 열어 시작합니다.

```bash
./build/lidar_viewer dataset/gangnam/2019_10_04_12_28_25.bin
```

## 실시간 LiDAR · IMU

프로그램을 먼저 실행한 뒤 상단 **Live... → Connect**를 누릅니다. 센서 드라이버는 별도로 실행돼 있어야 합니다. 기본 토픽은 XT32 `/b2/points`, IMU `/vn100/imu/data_ned`이며, 연결 창에서 변경할 수 있습니다. 뷰어가 센서 설정을 바꾸거나 로봇 구동 명령을 발행하지는 않습니다.

```bash
source /opt/ros/jazzy/setup.bash
./build/lidar_viewer --live --topic /b2/points \
  --imu-topic /vn100/imu/data_ned --sensor-height 0.57 --imu-level --2d
```

- **Pause / Space**: 표시 중인 프레임을 정지합니다. 센서 구독은 유지하며, 수신 횟수는 계속 증가합니다. 정지한 점군을 회전·확대하고 ROI를 수정할 수 있습니다.
- **Play / Space**: 쌓인 과거 프레임을 재생하는 대신 최신 프레임으로 복귀합니다. 화면용 최신 프레임은 하나만 보관해 정지 중 메모리가 계속 늘지 않습니다. 이전 시점의 재생이 필요하면 녹화한 bag을 엽니다.
- **Record / Stop REC**: 원본 LiDAR·IMU·`/tf`·`/tf_static`을 MCAP으로 저장하고 중지 시 `metadata.yaml`을 완성합니다. 화면 정지·ROI·지면분리·표시 좌표 보정은 원본 녹화 내용에 영향을 주지 않습니다. 녹화는 버튼을 눌렀을 때만 시작합니다.
- 저장 상위 폴더는 **Live...**에서 설정합니다. 소스 실행 기본값은 프로젝트 `recordings/`이며, `--record-folder /path/to/bags`로도 지정할 수 있습니다. 각 녹화는 한국 시간의 `sensors_YYYYMMDD_HHMMSS_microseconds` 폴더를 새로 만듭니다. 기존 폴더는 덮어쓰지 않습니다.
- **Live... → Disconnect**: 수신을 종료하고 진행 중인 녹화를 마무리합니다. 마지막 화면은 유지합니다. 파일을 열려면 먼저 연결을 해제합니다.
- 센서 QoS는 Best Effort/Volatile이며, 라이다 대기열은 두 프레임입니다. 표시가 늦어지면 과거 화면을 누적하지 않습니다. 녹화 큐는 별도 스레드에서 최대 256 MiB까지 보관하며, 디스크가 따라가지 못해 누락되면 횟수와 오류를 표시합니다. 센서·네트워크 단계의 무손실을 보장하는 기능은 아닙니다.
- 실시간 IMU도 수신 시각 차이 50ms 이내에서 중력을 대응합니다. 정지한 프레임에는 해당 프레임의 IMU 대응 결과가 고정됩니다. 실제 장착 회전과 센서 시간 동기화는 별도로 맞춰야 합니다.
- 같은 ROS domain/RMW 환경에서 실행해야 합니다. 예: 센서 실행 터미널과 뷰어 터미널의 `ROS_DOMAIN_ID`, 네트워크 검색 설정을 동일하게 사용합니다.

## 플랫폼별 빌드·배포

| 배포 | 지원 기능 | 실행 환경 |
| --- | --- | --- |
| Ubuntu 24.04 x86_64 `.deb`, `.tar.gz` | BIN·ROS bag·실시간 LiDAR/IMU·MCAP 녹화·ROI·지면분리 | ROS 2 Jazzy APT 저장소 및 런타임 |
| Windows x64 `.zip` | BIN·ROI·지면분리·네이티브 파일 선택 | 포함된 DLL과 `lidar_viewer.exe`를 같은 폴더에 유지 |
| macOS `.zip`, `.dmg` | BIN·ROI·지면분리·네이티브 파일 선택 | 패키지의 CPU 아키텍처에 맞는 macOS, OpenGL 4.1 |

배포 파일은 CPU 빌드입니다. CUDA는 소스에서 선택적으로 켭니다. Windows/macOS 배포에는 ROS 2가 포함되지 않아 ROS bag·실시간 ROS 연결은 비활성화됩니다. macOS 번들은 ad-hoc 서명이며 Apple Developer ID 서명·공증은 포함하지 않습니다.

- Ubuntu: ROS 2 Jazzy 저장소를 설정한 뒤 `sudo apt install ./LiDARViewer-*.deb`, 실행은 앱 메뉴의 **LiDAR Viewer** 또는 `lidar-viewer`입니다. 실행 스크립트가 Jazzy 환경을 읽습니다.
- Windows: ZIP을 풀고 `lidar_viewer.exe`를 실행합니다. **Open file**에서 `.bin`도 선택할 수 있습니다.
- macOS: DMG의 `lidar_viewer.app`을 Applications에 복사하거나 ZIP의 앱을 실행합니다.
- 패키지 빌드는 `-DLIDAR_VIEWER_USER_SETTINGS=ON`을 사용해 사용자 설정을 Windows `%APPDATA%/LiDARViewer`, macOS `~/Library/Application Support/LiDARViewer`, Ubuntu `~/.config/lidar_viewer`에 저장합니다. 기본 설정은 배포 리소스에서 읽고 사용자 설정으로 덮어씁니다. 업데이트해도 사용자 ROI를 패키지 설정으로 덮어쓰지 않습니다. `XDG_CONFIG_HOME`도 지원합니다.
- 소스 빌드는 기존처럼 프로젝트 `config/profiles`에 저장합니다. 사용자 경로를 쓰려면 위 CMake 옵션을 켭니다.

Windows 소스 빌드는 저장소의 `vcpkg.json`과 `cmake/triplets/x64-windows-release.cmake`를 사용합니다. macOS 의존성은 `brew install pcl glfw glew nanoflann ninja`로 설치합니다. 두 플랫폼 모두 기본 CI 설정은 `-DLIDAR_VIEWER_ENABLE_CUDA=OFF -DLIDAR_VIEWER_ENABLE_ROSBAG=OFF`입니다.

GitHub Actions의 **Build and release**는 PR·main push·수동 실행에서 세 플랫폼을 빌드하고 CTest와 Ubuntu 실시간 GUI 검사와 ROI 생성·편집·복구 검사를 실행한 뒤 패키지를 아티팩트로 올립니다. **`v*` 태그를 push하면 모든 플랫폼이 통과한 후에만 GitHub Release에 패키지와 SHA256SUMS를 게시합니다.** 릴리스용 비밀 토큰은 추가로 필요하지 않고, 해당 작업의 `GITHUB_TOKEN`에만 `contents: write` 권한을 부여합니다.

## 수집한 ROS 2 bag 열기

실행 중에는 상단 **Open file** 버튼 또는 **Ctrl+O**로 시스템 파일 선택 창을 열 수 있습니다. 선택하면 해당 bag을 열고 첫 프레임에서 정지합니다. 취소하거나 열기에 실패하면 현재 점군을 유지합니다. 단일 bag을 고르는 동작은 상위 폴더 전체를 등록하지 않습니다.

```bash
cd ~/workspace/lidar
source /opt/ros/jazzy/setup.bash
./build/lidar_viewer \
  ~/shalom_ws/data/bags/sensors_20261023_120204_157971 \
  --topic /b2/points --sensor-height 0.57 --play
```

### 폴더를 등록해서 선택하기

1. 하단 **Add folder**를 누릅니다.
2. 경로에 `~/shalom_ws/data/bags`를 입력하거나 폴더를 찾아 **Add this folder**를 누릅니다.
3. 하단 데이터 목록에서 날짜를 선택하거나 이름을 검색한 뒤 bag을 고릅니다.

폴더는 하위 폴더까지 검색하며, 같은 bag의 분할 파일이나 중복 등록은 하나로 표시합니다. 날짜는 수집 파일명 기준이며, 날짜를 알 수 없는 이름은 `Unknown date`로 표시합니다. 등록할 때 현재 데이터는 유지하고, 다른 bag을 선택하면 해당 bag만 색인한 뒤 첫 프레임에서 정지합니다. 열기에 실패하면 이전 데이터를 유지합니다.

등록 경로는 `~/.config/lidar_viewer/data_folders.cfg`에 저장합니다. `XDG_CONFIG_HOME`을 지정하면 해당 경로 아래에 저장합니다. 다음 실행 때 목록을 복원하지만 bag은 자동으로 열지 않습니다. **Open file**을 누르면 등록 폴더에서 탐색을 시작합니다. 새로운 bag을 같은 폴더에 넣었다면 폴더를 다시 추가하거나 앱을 재실행하면 목록을 갱신합니다. `--inspect`만 지정한 창 없는 실행은 등록 폴더의 최신 bag을 확인합니다.

폴더를 명령행에서 일회성으로 열 수도 있습니다.

```bash
./build/lidar_viewer ~/shalom_ws/data/bags --sensor-height 0.57
```

- bag 폴더, `metadata.yaml`, 개별 `.mcap`·`.db3` 파일을 직접 읽습니다. 원본을 변경하거나 `.bin`으로 변환하지 않습니다.
- 처음에는 선택한 토픽의 프레임 시각을 색인하고, 화면에 필요한 프레임만 읽습니다. 큰 bag은 최초 색인에 시간이 걸립니다.
- `--play` 없이 실행하면 첫 프레임에서 일시정지합니다. `Space`로 재생/정지, 타임라인으로 이동합니다. 재생 FPS는 기록 시각 간격의 중앙값으로 초기화하며, 기록 중 긴 공백을 실시간 그대로 재현하는 플레이어는 아닙니다.
- `--sensor-height 0.57`은 측정한 라이다 높이에 따른 **표시용 Z +0.57 m**입니다. 생략하면 별도 `rosbag` 프로파일의 보정값을 사용합니다(초기값 0). bag에는 기존 BIN의 자동 지면 높이 이동을 적용하지 않습니다.
- `rosbag` 프로파일은 기존 Gangnam/KITTI의 회전·높이 설정을 사용하지 않습니다. 처음에는 노면 제거·클러스터링이 꺼져 있으므로 원시 점군을 본 뒤 `Ground` 탭에서 켜서 비교할 수 있습니다.
- PointCloud2의 필드 offset·26바이트 stride·행 padding·endian을 읽어 XYZI를 추출합니다. 반사 강도는 읽기 단계에서 기존 방식대로 프레임별 정규화하며, 색상 표현은 기존 뷰어 설정을 따릅니다. IMU/odom 기반 deskew·SLAM은 수행하지 않습니다.
- ROS 노드나 DDS 통신 없이 오프라인으로 읽으므로 `ros2 bag play`, 센서 드라이버, RViz는 필요 없습니다.

창 없이 입력을 확인하려면 같은 명령에서 `--play` 대신 `--inspect`를 사용합니다.

### IMU 자동 수평 보정

`Align`은 현재 상태를 **MANUAL / MANUAL FALLBACK / AUTO · IMU ACTIVE**로 표시합니다. IMU가 없거나 장착 회전·시각 대응을 사용할 수 없으면 자동 보정을 요청한 상태에서도 수동 pitch/roll을 입력할 수 있습니다. 자동 보정이 실제 적용될 때는 입력 칸에 **실제로 적용된 IMU pitch/roll**을 보여 주며, 기존 수동 각도는 보존합니다. 자동 보정을 끄면 그 수동 각도로 돌아갑니다. 각도와 높이는 직접 입력하거나 +/- 버튼으로 조절합니다.

**Reset rotation**은 IMU 자동 보정을 끄고 표시 yaw/pitch/roll을 0도로 돌리며, 센서 높이를 유지합니다. **Reset all**은 높이까지 0으로 돌립니다. 두 동작 모두 센서 장착 설정을 바꾸지 않습니다.

**Sensor mounting (advanced)**는 IMU 센서축을 LiDAR 센서축으로 바꾸는 설정입니다. 화면의 작은 기울기를 조정하는 값과 구분하세요. 원본 VN100 NED/FRD 벡터와 FLU XT32 조합에는 roll 180° 변환이 필요합니다. 이미 어댑터에서 FLU로 변환한 데이터에는 이를 다시 적용하면 안 됩니다. 해당 센서 조합에는 **Raw NED -> FLU / Prepared FLU** 프리셋을 제공합니다. 원본 NED 데이터와 작은 장착 roll 값의 조합으로 약 180° 보정이 계산되면 상태 영역에 축 변환 버튼을 표시합니다. 저장된 값을 자동으로 덮어쓰지는 않으며, 프리셋을 적용한 뒤 **Save profile**로 저장합니다. 정적 TF가 있으면 그 출처를 표시하고 사용하지 않는 프로파일 장착 입력을 잠급니다.

왼쪽 **Align → IMU auto level**을 켜면 기록된 IMU의 중력 방향이 표시 좌표계의 +Z가 되도록 roll/pitch를 보정합니다. 보정한 점군을 지면 분할에도 사용하며 yaw와 센서 높이는 기존 설정을 유지합니다. 실제 경사로는 경사로로 남고, 지형을 평면에 강제로 맞추지는 않습니다.

```bash
./build/lidar_viewer ~/shalom_ws/data/bags --topic /b2/points \
  --sensor-height 0.57 --imu-level --2d
```

- IMU 자세 quaternion이 유효하면 사용하고, 자세 정보가 없으면 가속도·자이로로 중력 방향을 추정합니다. 가속도 크기가 중력과 크게 다르거나 회전이 빠르면 가속도에 의한 수정을 억제합니다. 큰 시각 공백과 오래된 추정값은 사용하지 않습니다.
- 점군과 IMU는 같은 **bag 수신 시각**으로 대응하며 최대 50ms 차이까지만 사용합니다. 이는 센서 하드웨어 시간 동기화나 포인트별 deskew가 아닙니다. IMU 기록이 시작되기 전 구간(예시 VN100 bag에서는 약 2초)에는 보정을 적용하지 않습니다. 기록 자체가 없는 경우와 중력 추정이 불안정한 경우를 구분해 표시합니다.
- IMU→LiDAR 회전은 기록된 `/tf_static` 경로를 우선 사용합니다. 경로가 없으면 명시한 프로파일의 센서 프레임 조합과 장착 회전을 사용합니다. 그 외에는 `Sensor mounting (advanced)`에서 장착 회전을 설정·확인해야 합니다.
- 기본 rosbag 프로파일에는 기존 사용자 제공 근삿값에 따라 `vn100_link_ned → pandar_xt32`의 FRD→FLU 변환(roll 180°, pitch/yaw 0°)이 들어 있습니다. 다른 프레임에는 적용하지 않습니다. 이 설정은 정밀 외부 보정값이 아닙니다.
- 현재 VN100 원본 토픽의 quaternion은 단위값으로 고정돼 있어 자세 추정으로 사용하지 않습니다. `/vn100/imu/data_ned`가 있으면 이를 우선 선택해 가속도·자이로를 사용합니다. 다른 IMU 토픽은 `--imu-topic /your/imu`로 선택할 수 있습니다.
- IMU·장착 회전·시각 대응이 불충분하면 이유를 표시하고 기존 수동 보정을 유지합니다. 자동 보정 토글과 장착 회전은 `Save profile`로 저장합니다.

## Controls

- 왼쪽 `View`: 2D/3D, 전체 맞춤·센서 시점, 색상·점 크기, 격자, UI 배율
- 왼쪽 `Ground`: 노면 분할 방식·해당 알고리즘의 셀 크기, 클러스터링, 실험용 형상 라벨
- 왼쪽 `Align`: 표시용 yaw/pitch/roll/Z. 원본 데이터는 바꾸지 않습니다.
- 왼쪽 `ROI`: 여러 3D 박스 추가·수정·복제·삭제, 박스 내부 점만 표시
- 하단: 폴더 추가, 날짜·이름별 bag 선택, 재생·정지, 프레임 이동, 반복, 재생 FPS, 타임라인. 드래그 중 일시정지하며 놓을 때 선택한 프레임을 읽습니다.
- `Save profile`: 현재 프로파일에 명시적으로 저장합니다. 저장 실패는 오류로 표시하고 기존 파일을 보존합니다.

- `Left drag`: 3D orbit / 2D pan
- `Right drag`: 3D pan / 2D rotation
- `Q` / `E`: 2D 회전. `View`의 회전 슬라이더·초기화 버튼도 지원합니다. 2D/3D 시점은 각각 저장합니다.
- `Wheel`: zoom
- `N` or `Right`: next frame
- `P` or `Left`: previous frame
- `F`: fit view
- `C`: intensity / height / uniform color cycle
- `[` `]`: point size
- 숫자 직접 입력: 슬라이더를 `Ctrl+click`한 뒤 입력·`Enter`
- `Esc`: quit
- `Ctrl+O`: 시스템 파일 탐색 창에서 bag 열기

## 3D ROI

- **화면에서 생성**: `R` 또는 캔버스의 **Draw box**를 누릅니다. 2D 시점으로 전환된 뒤 마우스로 XY 영역을 드래그하면 박스가 만들어집니다. `Creation settings`의 중심 Z·높이로 3D 범위를 정합니다. 박스의 yaw는 현재 2D 화면의 가로 방향을 따릅니다. 반대 방향으로 드래그해도 됩니다. `Esc`는 그리기를 취소하며, 너무 작은 클릭은 박스로 만들지 않습니다.
- **이동**: **Move** 또는 **Move in 2D**를 누르고 박스를 드래그합니다. 높이·yaw는 유지됩니다. **Navigate** 또는 `Esc`로 카메라 조작으로 돌아갑니다. 회전한 2D 시점과 IMU 수평 보정에서도 원본 라이다 좌표로 대응합니다.
- **숫자로 생성·편집**: ROI 탭의 **+ Box**는 현재 시점 중심에 박스를 만듭니다. 중심 XYZ(m), 전체 크기 XYZ(m), yaw(도)를 각 칸에 바로 입력할 수 있습니다. yaw는 숫자 입력만 사용하며 **Reset**으로 0도로 돌립니다. `Creation settings`에서 20×20m·4×4m·센서 전방 12×8m 프리셋을 선택할 수 있습니다.
- **목록·복제·복구**: 목록의 체크박스는 개별 박스를 켜고 끕니다. **Duplicate / Ctrl+D**는 선택 박스의 로컬 X 방향으로 0.5m 간격을 두고 복제합니다. **Delete / Delete 키**로 삭제하며 **Undo / Ctrl+Z**, **Redo / Ctrl+Y**로 최근 64개 편집을 복구할 수 있습니다. 숫자 한 번의 편집과 드래그 한 번을 하나의 작업으로 취급합니다. 텍스트 입력 중에는 해당 입력 칸의 실행 취소를 사용합니다.
- **표시·저장**: **Crop points**는 활성 박스들의 합집합 안에 있는 점만 표시합니다. 겹친 점은 중복되지 않으며 활성 박스가 없으면 빈 점군이 됩니다. **Outlines**는 박스 외곽선을 표시하고 선택한 박스는 노란색입니다. **Focus box**로 선택한 박스에 시점을 맞춥니다. **Save profile / Ctrl+S**로 저장하며 저장하지 않은 ROI 변경은 하단에 표시됩니다. `Coordinates & corners`에서 8개 꼭짓점을 복사할 수 있습니다. 최대 128개 박스입니다.

좌표는 **IMU·수동 회전·Z 보정 전 원본 라이다 좌표계** 기준입니다. 박스를 드래그하는 동안에는 외곽선을 미리 보고 놓을 때 필터를 적용합니다. ROI를 수정하면 표시 재생을 정지하며, 실시간 구독과 녹화는 계속됩니다.

지면분리는 전체 프레임으로 계산해 캐시합니다. ROI 편집은 캐시를 필터링하고 ROI 안의 장애물만 클러스터링하므로 주변 지면 정보와 분류 결과가 유지됩니다. 새 프레임이나 보정·지면분리 설정 변경 시에는 캐시를 다시 만듭니다. 상태 표시줄에서 전체 처리와 ROI 처리 시간을 확인할 수 있습니다. 격자와 좌표축도 별도 GPU 버퍼를 사용해 정지 화면에서 매 프레임 다시 전송하지 않습니다.

## Notes

- KITTI 스타일 `float32 x, y, z, intensity` `.bin` 파일을 바로 지원합니다.
- Ouster `OS1-64 LEGACY` 레코드 컨테이너도 지원합니다.
- dataset profile은 입력 경로로 자동 감지됩니다. `gangnam`, `kitti`, `rosbag`, `default` 프로파일을 지원합니다.
- 저장된 설정은 프로젝트의 `config/profiles/<profile>.cfg`에 기록되고 다음 실행 때 자동으로 불러옵니다. 다른 폴더에서 실행해도 실행 파일 위치와 빌드 시 프로젝트 경로로 같은 설정·기본 데이터를 찾습니다. 프로파일이 없을 때의 좌표 보정 기본값은 회전 0, Z 0입니다.
- 노면 분할 토글·알고리즘과 셀 크기도 설정 파일에 함께 저장됩니다.
- calibration rotation과 z translation 값도 같은 설정 파일에 함께 저장됩니다.
- 기본값으로 모든 유효 포인트를 GPU에 올려서 그립니다. NaN/Inf 좌표 샘플은 BIN과 bag에서 동일하게 제외합니다.
- 노면 분할 기본값은 `fast_grid`입니다. 기존 CUDA 격자 경로의 규칙을 CPU에도 구현해 CUDA를 끄거나 사용할 수 없을 때 같은 방식으로 분류합니다. 원본 GSeg3D 라이브러리는 별도 `gseg3d` 모드(`GSeg3D (reference CPU)`)로 선택하며 CUDA 토글에 영향을 받지 않습니다. 기존 프로파일에 `ground_mode=gseg3d`를 저장했다면 기준 CPU 구현을 선택합니다. 이전 CUDA 경로를 계속 사용하려면 UI에서 `Fast grid`를 고르고 프로파일을 저장합니다.
- GSeg3D는 ROS2 wrapper 없이 핵심 라이브러리만 `third_party/ground_segmentation` submodule로 포함합니다. `patchworkpp`, `grid`, `ransac` 비교 모드도 사용할 수 있습니다.
- 클러스터링은 실제 점 간 수평 거리와 수직 간격을 검증하는 range-adaptive Euclidean 방식입니다. 작은 공간 해시 셀·거리별 검색 범위를 사용하고, 이미 연결한 점은 후보에서 제거해 고밀도 점군의 반복 검사를 줄입니다. 회전 박스의 긴 축을 길이로 정규화하며 CUDA 작업 버퍼는 프레임 간 재사용합니다.
- 형상 라벨은 클러스터 치수 기반 휴리스틱입니다. 학습된 사람·차량 인식기로 간주하면 안 됩니다.

## Test

```bash
/usr/bin/ctest --test-dir build --output-on-failure
```

실시간 C++ 검사는 별도 ROS domain에서 합성 LiDAR/IMU를 발행해 최신 프레임 교체·정지 프레임 보존·IMU/TF 대응·원본 MCAP 저장·중지/연결 해제 시 메타데이터 완성을 검사합니다. 실제 장비 드라이버를 실행하지 않습니다.

```bash
xvfb-run -a -s '-screen 0 1920x1200x24' /usr/bin/python3 tests/ui_live_smoke.py \
  --viewer build/lidar_viewer --output build/test_artifacts/live
```

알고리즘 회귀 검사는 음수 좌표, 물체 회전, 완전 탐색과의 클러스터 연결성 비교, 고밀도 입력, CPU/CUDA fast-grid 결과 일치와 버퍼 재사용을 확인합니다. 설정·입력 검사는 실행 디렉터리 독립성, 잘못된 숫자와 NaN/Inf BIN 샘플을 검사합니다.
IMU 검사는 ENU/NED, 센서 장착 회전, 중력 추정, 오래된 샘플 거부, 정적 TF 합성, 경사 지형 보존과 회전한 2D 시점의 이동을 검사합니다.
ROI 검사는 yaw·3D 경계, 다중 박스 합집합, IMU/표시 변환 시 선택 영역 유지, 설정 왕복과 잘못된 설정 거부를 검사합니다.
파일 선택 창 검사는 특수문자가 있는 경로의 전달, 취소·실패 처리, 실행 중인 선택 창의 종료를 검사합니다.

OpenGL 입력·배율·크기 변경 회귀 테스트는 별도 Xvfb에서 실행합니다. 실제 데스크톱의 입력이나 저장된 설정은 변경하지 않습니다.

```bash
xvfb-run -a -s '-screen 0 1920x1200x24' /usr/bin/python3 tests/ui_gl_smoke.py \
  --viewer build/lidar_viewer \
  --bag ~/shalom_ws/data/bags/sensors_20261023_120204_157971 \
  --output build/test_artifacts/ui_redesign --dpi 192
```

2D 회전·IMU 토글·프로파일 저장 검사는 별도 실행 파일 복사본과 설정 폴더를 사용합니다.

```bash
xvfb-run -a -s '-screen 0 1920x1200x24' /usr/bin/python3 tests/ui_imu_rotation_smoke.py \
  --viewer build/lidar_viewer \
  --bag ~/shalom_ws/data/bags/sensors_20261022_192513_438096 \
  --output build/test_artifacts/imu_rotation
```

ROI 편집·다중 박스·복제/삭제·저장/복원·표시 변환·지면분리 연동 검사는 합성 점군과 별도 설정 폴더를 사용합니다. IMU GUI 검사에 `--roi`를 추가하면 실제 bag의 ROI와 자동 수평 보정을 함께 검사합니다.

```bash
xvfb-run -a -s '-screen 0 1920x1200x24' /usr/bin/python3 tests/ui_roi_smoke.py \
  --viewer build/lidar_viewer --output build/test_artifacts/roi
```

빈 뷰어 시작·자동 로딩 방지·빈 상태의 재생/프레임 조작·버튼으로 파일 탐색 창 열기·취소·분할 bag 전환은 별도 Xvfb에서 검사합니다.

```bash
xvfb-run -a -s '-screen 0 1920x1200x24' /usr/bin/python3 tests/ui_file_dialog_smoke.py \
  --viewer build/lidar_viewer \
  --bag ~/shalom_ws/data/bags/sensors_20261022_192513_438096 \
  --target ~/shalom_ws/data/bags/sensors_20261023_120204_157971 \
  --output build/test_artifacts/file_dialog
```
