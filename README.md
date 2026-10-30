# LiDAR OpenGL Viewer

CMake + OpenGL 기반 LiDAR 뷰어입니다. `.bin`과 ROS 2 PointCloud2 bag을 지원합니다.

## Build

```bash
cd ~/workspace/lidar
source /opt/ros/jazzy/setup.bash
git submodule update --init --recursive
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
- `src/algorithms/`: 노면 제거, 3D/XY 클러스터링, 학습 모델 박스 변환
- `src/learned_detector.cpp`, `python/lidar_detection/`: 비동기 CenterPoint ONNX 추론
- `include/lidar_viewer/app.hpp`: 앱 진입 선언
- `include/lidar_viewer/algorithms/`: 알고리즘 인터페이스와 공통 타입

## Run

입력 경로를 생략하면 **빈 뷰어가 먼저 열립니다.** 상단 **Open file** 버튼을 누르면 시스템 파일 탐색 창이 열립니다. bag 폴더 안의 `metadata.yaml` 또는 `.db3`·`.mcap` 파일을 선택하세요. 분할 파일을 골라도 같은 폴더의 기록 전체를 엽니다. 탐색 창을 취소해도 뷰어는 계속 열려 있습니다. bag을 열기 전에는 재생·프레임 이동·타임라인이 비활성화됩니다.

```bash
source /opt/ros/jazzy/setup.bash
./build/lidar_viewer --sensor-height 0.57 --imu-level --2d
```

BIN 파일이나 단일 bag 폴더를 뷰어에 드래그해 열 수도 있습니다. 하단은 소스 선택·폴더 등록·재생·타임라인에 사용하고, 캔버스 상단은 2D/3D 전환에 사용합니다. ROI 탭에서 Edit·Draw·Undo·Redo와 박스 생성·복제·삭제를 함께 관리합니다.

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

실행 중에는 상단 **Open file** 버튼 또는 **Ctrl+O**로 시스템 파일 선택 창을 열 수 있습니다. 선택하면 해당 bag을 열고 첫 프레임에서 정지합니다. 취소하거나 열기에 실패하면 현재 점군을 유지하며, 재생 중이었다면 재생도 복원합니다. 로딩 카드에는 파일명·색인한 LiDAR 프레임/전체 수·IMU 표본 수·경과 시간·선택 메시지의 스캔 진행률을 표시합니다. 전체 수를 아직 모르는 단계는 활동 표시줄로 구분합니다. **Cancel opening / Esc**로 취소할 수 있으며, 명령행에서 bag을 열어도 창이 먼저 표시됩니다. 단일 bag을 고르는 동작은 상위 폴더 전체를 등록하지 않습니다.

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

### 지면만 표시하기

**Ground → Show → Ground only**를 선택하면 ROI 안의 지면 포인트만 표시합니다. **All points / Obstacles only**로 전체·장애물만 표시할 수도 있습니다. 제한된 표시 모드를 선택하면 지면분리를 자동으로 켜고, 분류를 끄면 전체 표시로 복귀합니다. **Save profile**로 선택을 저장합니다.

분류 결과의 GPU 그리기 범위만 바꾸므로 표시 전환 때문에 지면분리·클러스터링을 다시 실행하거나 재생을 정지하지 않습니다. Ground only에서는 장애물 박스도 숨깁니다. 원본 데이터와 분석 결과는 유지합니다.

## Controls

- 왼쪽 `View`: 색상·점 크기·격자·카메라 설정. UI 배율은 오른쪽 위 앱 설정에서 관리합니다.
- 왼쪽 `Ground`: 지면분리 방식·각 방식에 해당하는 파라미터
- 왼쪽 `Objects`: 학습 모델 검출 방식·검출 대상 또는 기하학 클러스터링
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

캔버스 도구 모음은 **2D/3D** 전환에 사용하고, **Edit / Draw / Undo / Redo**는 ROI 탭에서 관리합니다. 왼쪽은 **View**(표시), **Align**(좌표·IMU), **Ground**(지면분리), **ROI**(영역), **Objects**(학습 모델 검출·클러스터)로 구분합니다. 프로그램 UI 배율은 오른쪽 위 톱니바퀴의 **Application settings**에서 관리합니다. 배율은 사용자 설정 폴더의 `app_settings.cfg`에 저장하며, 데이터 프로파일이나 bag을 바꿔도 유지합니다. 기존 프로파일의 배율은 공통 설정이 아직 없을 때 최초 배율로만 사용합니다. ROI 때문에 표시할 점이 없을 때는 로딩 중 상태와 구분해 빈 영역 안내와 전체 점군 보기 동작을 제공합니다.

- **만들기**: ROI 탭의 **Draw** 또는 **R**을 누른 뒤 대각선으로 드래그합니다. **New** 아이콘은 현재 시점 중심에 기본 4×4m 박스를 만듭니다. `New box defaults`에서 새 박스의 높이와 프리셋을 바꿉니다.
- **직접 편집**: **Edit**에서 박스 내부를 끌면 이동하고, 8개 모서리·변 핸들을 끌면 반대편을 고정한 채 크기를 바꿉니다. 바깥의 원형 핸들은 yaw를 회전하며 Shift를 누르면 15도 단위로 맞춥니다. 빈 공간 드래그는 화면 이동입니다.
- **높이·정밀 값**: ROI 탭의 **Bottom / Top**을 마우스로 끌어 아래·윗면을 조절합니다. Ctrl+클릭하면 숫자를 입력합니다. `Exact geometry`를 펼치면 이름·중심 XYZ·전체 크기 XYZ·yaw를 직접 입력할 수 있습니다. yaw 슬라이더는 없습니다.
- **관리**: 목록에서 선택하고, 선택 박스를 화면에 맞추기·복제·삭제 아이콘을 사용합니다. 목록을 더블클릭해도 해당 박스에 맞춥니다. 복제는 로컬 X 방향으로 0.5m 간격을 두어 생성합니다. ROI 탭의 되돌리기/다시 실행 아이콘 또는 **Ctrl+Z / Ctrl+Y**로 복구합니다.
- **표시·저장**: **Crop points**는 활성 박스의 합집합 안에 있는 점만 표시합니다. 눈 아이콘은 외곽선을 전환합니다. **Save profile / Ctrl+S**로 저장하며 `Export corners`에서 8개 꼭짓점을 복사합니다. 최대 128개 박스입니다.

좌표는 IMU·수동 회전·Z 보정 전 원본 LiDAR 좌표계입니다. 도구 선택, 수치 수정, 복제·삭제와 일반 UI 조작은 재생을 멈추지 않습니다. 실제 박스 드래그 중에만 화면을 고정하고, 놓거나 취소하면 이전 재생 상태로 복귀합니다. 사용자가 Pause한 화면은 계속 정지 상태를 유지합니다. 실시간 구독과 원본 녹화는 계속됩니다.

지면분리는 전체 프레임으로 계산해 캐시합니다. ROI 편집은 캐시를 필터링하고 ROI 안의 장애물만 클러스터링하므로 주변 지면 정보와 분류 결과가 유지됩니다. 파일 읽기와 지면분리·클러스터링은 각각 재사용하는 작업 스레드에서 실행합니다. 첫 프레임 준비도 백그라운드에서 처리합니다. 작업 중 설정을 바꾸면 오래된 결과를 적용하지 않고 새 설정을 사용합니다. 대기 작업은 최신 하나만 유지하며, 읽기 캐시는 최대 3프레임·64 MiB로 제한합니다. 단일 프레임 자체가 64 MiB보다 크면 그 프레임 하나만 유지합니다. UI와 ROS 구독·녹화는 분석을 기다리지 않습니다.

## Notes

- KITTI 스타일 `float32 x, y, z, intensity` `.bin` 파일을 바로 지원합니다.
- Ouster `OS1-64 LEGACY` 레코드 컨테이너도 지원합니다.
- dataset profile은 입력 경로로 자동 감지됩니다. `gangnam`, `kitti`, `rosbag`, `default` 프로파일을 지원합니다.
- 저장된 설정은 프로젝트의 `config/profiles/<profile>.cfg`에 기록되고 다음 실행 때 자동으로 불러옵니다. 다른 폴더에서 실행해도 실행 파일 위치와 빌드 시 프로젝트 경로로 같은 설정·기본 데이터를 찾습니다. 프로파일이 없을 때의 좌표 보정 기본값은 회전 0, Z 0입니다.
- 노면 분할 토글·알고리즘과 셀 크기도 설정 파일에 함께 저장됩니다.
- calibration rotation과 z translation 값도 같은 설정 파일에 함께 저장됩니다.
- 기본값으로 모든 유효 포인트를 GPU에 올려서 그립니다. NaN/Inf 좌표 샘플은 BIN과 bag에서 동일하게 제외합니다.
- 노면 분할 기본값은 `fast_grid`입니다. 기존 CUDA 격자 경로의 규칙을 CPU에도 구현해 CUDA를 끄거나 사용할 수 없을 때 같은 방식으로 분류합니다. 원본 GSeg3D 라이브러리는 별도 `gseg3d` 모드(`GSeg3D`)로 선택하며 CUDA 토글에 영향을 받지 않습니다. 기존 프로파일에 `ground_mode=gseg3d`를 저장했다면 기준 CPU 구현을 선택합니다. 이전 CUDA 경로를 계속 사용하려면 UI에서 `Fast grid`를 고르고 프로파일을 저장합니다.
- GSeg3D는 ROS2 wrapper 없이 핵심 라이브러리만 `third_party/ground_segmentation` submodule로 포함합니다. `patchworkpp`는 [KAIST 원본 Patchwork++](https://github.com/url-kaist/patchwork-plusplus) submodule의 C++ 구현을 사용합니다. 원본의 R-VPF·적응형 GLE·TGR을 사용하고, 데이터/탐색 위치/파라미터가 바뀌면 적응 이력을 초기화합니다. 프레임별로 정규화된 강도를 원시 반사 강도로 오인하지 않도록 RNR은 끕니다. 알고리즘 입력은 보정된 센서 높이를 기준으로 변환하고, 반환 인덱스로 원본 점을 분류해 중복점·색상·범위 밖 점도 보존합니다. `grid`와 `ransac`은 각각 연결 격자와 전역 평면에 기반한 구현입니다.
- 클러스터링은 실제 점 간 수평 거리와 수직 간격을 검증하는 range-adaptive Euclidean 방식입니다. 작은 공간 해시 셀·거리별 검색 범위를 사용하고, 이미 연결한 점은 후보에서 제거해 고밀도 점군의 반복 검사를 줄입니다. 회전 박스의 긴 축을 길이로 정규화하며 CUDA 작업 버퍼는 프레임 간 재사용합니다.
- 기하학 클러스터는 장애물 점의 그룹만 표시합니다. 사람·차량 클래스는 학습된 CenterPoint 모델만 제공합니다.

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

### 지면분리 검증 범위

GSeg3D·Patchwork++·FugSeg는 원본 라이브러리를 연결하며, Fast grid·Grid·RANSAC도 실제 분할을 수행합니다. 지면분리를 끈 상태에서 방식을 선택하면 분류를 켜서 결과가 바로 반영됩니다. Patchwork++에는 해당 방식이 사용하지 않는 격자 크기·CUDA 조작을 표시하지 않습니다.

CTest는 여섯 방식 모두에 대해 정답이 정해진 평지·약 7도 경사와 45cm 이상 높이의 장애물 장면에서 지면 precision ≥98%, recall ≥85%, 전체 점 보존을 검사합니다. 이 제한된 합성 검사는 모든 실제 지형의 정확도를 보장하지 않습니다. GSeg3D의 높이 연속성 기본값은 원본 코어와 같은 0.30m이며, 이전 0.10m 값은 1m 셀의 경사 연결을 끊는 문제가 있었습니다. Ground reference에서 연속성·기준 높이를 조절할 수 있습니다.

실제 bag의 1/4·1/2·3/4 프레임에서 여섯 방식의 처리 시간과 분류 수를 비교하려면 다음을 실행합니다. 출력의 분류 수는 정답 라벨이 없는 데이터의 정확도 점수가 아닙니다. 아래 VN100 NED → XT32 장착 변환과 0.57m 높이는 제공된 원본 bag에 대한 값입니다.

```bash
./build/algorithm_smoke_test /path/to/raw_vn100_xt32_bag 0.57
```

재생 중 버튼·ROI 조작과 앱 공통 설정 검사는 별도 설정·합성 bag을 사용합니다.

```bash
xvfb-run -a -s '-screen 0 1920x1200x24' /usr/bin/python3 tests/ui_playback_smoke.py --viewer build/lidar_viewer --output build/test_artifacts/playback
xvfb-run -a -s '-screen 0 1920x1200x24' /usr/bin/python3 tests/ui_app_settings_smoke.py --viewer build/lidar_viewer --output build/test_artifacts/app-settings
```

## 사람·차량 인식 — 원본 CenterPoint

**Objects → Enable objects → Method**에서 다음 방식을 선택합니다.

- **CenterPoint tiny (learned)**: [CenterPoint 논문(CVPR 2021)](https://arxiv.org/abs/2006.11275)에 기반한 [Autoware의 원본 CenterPoint tiny v4.1 모델](https://huggingface.co/AutowareFoundation/lidar_centerpoint)입니다. **Targets**에서 Person·Bicycle·Car·Truck·Bus를 각각 켜고 끌 수 있고 All/Clear로 전체 선택·해제합니다. 대상과 Min score는 저장된 모델 결과를 즉시 필터링하며, 모델 로딩이나 추론을 다시 하지 않습니다. 선택은 프로파일에 저장합니다.
- **Euclidean 3D (clusters)**: 지면분리 후 가까운 장애물 점을 3차원 거리로 묶습니다. 사람·차량 클래스는 만들지 않습니다.
- **Projected XY (clusters)**: XY 거리와 수직 간격으로 묶습니다. 높이가 떨어진 면을 합칠 수 있으므로 보통 Euclidean 3D부터 사용합니다.

학습 모델은 **지면 제거·ROI 이전의 전체 수평 보정 스캔**을 받습니다. 클러스터가 잘못 합쳐져도 모델 입력이 잘리지 않습니다. CenterPoint 선택 시 **Cluster overlay**에서 기하학 그룹을 함께 볼 수도 있습니다.

이 컴퓨터의 GPU 런타임과 모델은 설치·실행 검증을 마쳤습니다. 기본 뷰어 실행 후 Objects에서 켜면 됩니다.

```bash
source /opt/ros/jazzy/setup.bash
./build/lidar_viewer --detect
```

다른 컴퓨터에서는 Python 3.10–3.13으로 선택 기능을 한 번 설치합니다. 시스템 Python 패키지를 바꾸지 않고 사용자 폴더의 별도 가상환경에 설치합니다. Ubuntu에서 venv가 없다면 `python3-venv` 패키지가 필요합니다.

```bash
# CPU: Ubuntu / Windows / macOS
python3 tools/setup_detector.py
# NVIDIA GPU: Linux / Windows x64, CUDA 12 + cuDNN 9 런타임을 별도 환경에 설치
python3 tools/setup_detector.py --cuda
# Windows에서는 위 python3 대신 py -3.12 사용 가능
```

배포본에도 `python/`과 `tools/setup_detector.py`가 포함됩니다. Ubuntu 설치본의 setup 경로는 `/usr/share/lidar_viewer/tools/setup_detector.py`, macOS는 앱의 `Contents/Resources/tools/setup_detector.py`입니다. 설치 후 실행 중인 앱에서는 **Retry model**을 누르거나 Enable objects를 켭니다. CUDA가 실행되지 않으면 실제 사용 중인 CPU 백엔드를 표시합니다.

모델 두 파일(약 20 MB)은 고정 revision `494c8171def40bd36cc2feb323e0a5acbfab132b`에서 내려받고 각각 SHA-256을 검증합니다. 원본 모델 카드도 함께 저장합니다. 기본 Linux 위치는 `~/.local/share/lidar_viewer/detector/`, 연결 설정은 `~/.config/lidar_viewer/detector.cfg`, 오류 로그는 같은 설정 폴더의 `detector.log`입니다. Windows는 LocalAppData/APPDATA, macOS는 Application Support를 사용합니다. 모델·가상환경은 Git과 배포 파일에 넣지 않습니다.

- **Align**에서 지면의 기울기와 높이를 먼저 맞춥니다. 모델 입력은 ±76.8 m XY, Z −4~6 m 범위이며 표시 yaw는 추론에서 되돌린 후 박스에 다시 적용합니다.
- 이 모델의 네 번째 입력은 intensity가 아닌 **시간 차이**입니다. 현재 스캔 하나의 시간 차이 0을 사용하며, 위치 추정 없이 이전 스캔을 억지로 합치지 않습니다. XT32의 미수신 XYZ=0 반환도 높이를 이동하기 전에 제외합니다.
- 원본 32-point pillar 특징, 9개 encoder 입력, 480×480 BEV와 원본 가중치를 유지합니다. 표준 ONNX ScatterND로 encoder와 head를 연결해 TensorRT 플러그인·별도 CUDA 컴파일 없이 실행합니다. 샘플 선택은 일시정지 시 결과가 흔들리지 않도록 결정적입니다. 40,000 pillar 상한을 넘으면 가까운 pillar를 우선하고 잘린 수를 처리 시간 툴팁에 표시합니다.
- 원본 형식의 크기·높이·yaw를 ROS 축으로 변환하고 circle NMS와 회전 BEV IoU NMS로 중복 박스를 억제합니다. 박스 회전과 클래스·점수를 표시하며 겹치는 글자는 숨깁니다. **Min score**는 신경망 분류 점수로, 보정된 확률이 아닙니다. 점수 0.1 이상의 후보를 캐시하므로 슬라이더 조절은 모델을 다시 실행하지 않고 즉시 반영됩니다.
- 모델 로딩과 추론은 별도 프로세스/작업 스레드에서 실행합니다. 표시할 스캔과 같은 revision의 박스만 적용합니다. 준비 후 추론이 재생보다 느리면 표시 프레임을 기다리지만 UI와 원본 ROS 구독·녹화는 계속 동작합니다. 초기 로딩·오류·2초 이상 지연은 재생을 막지 않습니다. ROI 편집은 저장된 검출의 중심점을 ROI로 필터링합니다.
- 모델은 한 프로세스와 한 ONNX 세션을 계속 사용합니다. 방식 변경·해제/재활성화·대상 선택은 프로세스를 재시작하지 않습니다. 명시적인 Retry나 실패 복구만 재시작합니다. 검출 입력은 불변 스냅샷으로 공유하고 XYZ 변환·IPC 버퍼는 작업 스레드에서 재사용합니다. 지면 파라미터나 클러스터 거리만 바꾸면 학습 검출을 다시 실행하지 않습니다.
- 중복 박스 억제는 원래 10 m 비교 범위를 공간 해시로 좁혀 같은 판정을 유지합니다. CUDA 라이브러리는 [ONNX Runtime의 사전 로딩 API](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls)를 사용하며 PyTorch 전체를 불러오지 않습니다.

2026-10-30, RTX 4060 Laptop / ORT 1.23.2, KITTI drive 0005의 41번 스캔(122,359개 유효 범위 점), 후보 점수 0.1, 준비 4회 후 30회 반복 측정: 변경 전후 검출 프로세스 RSS **1,142 → 719 MiB**, 모델 초기화 **2.92 → 0.48초**, 전체 파이프라인 p50 **69.3 → 62.9 ms**였습니다. 장비·입력·동시 작업에 따라 달라집니다. 원본 모델 그래프와의 수치 비교 및 NMS 완전 탐색 비교를 별도로 검사합니다. 현재 버전의 반복 시간과 메모리 변화는 설치된 검출용 Python으로 재측정할 수 있습니다.

추가 최적화에서는 XYZ 배열을 여러 번 복사하지 않고 인덱스만 정렬합니다. 빌드에 포함된 작은 C++ 라이브러리가 안정 계수 정렬과 9개 특징 생성을 처리하며, 원래의 셔플 순서·pillar당 32점·float32 연산 순서·패딩을 유지합니다. 모델 가중치·입력 특징·박스 억제 기준은 동일합니다. Python/NumPy C ABI나 CUDA에 의존하지 않으며 Windows·Ubuntu·macOS 패키지에 포함합니다. 사용할 수 없을 때는 NumPy 구현으로 동작합니다. 실제 사용 경로는 처리 시간 툴팁과 벤치마크의 `preprocess_backend`에서 확인할 수 있습니다.

같은 PC에서 이전 `18f478e`와 추가 최적화를 **같은 ONNX 세션에서 교차 측정**했습니다. 5회 준비 후 30회, 후보 점수 0.1의 p50입니다.

| 점군 | 전처리 | 후처리 | 전체 검출 파이프라인 |
| --- | ---: | ---: | ---: |
| KITTI 41번, 122,490점 | 30.8 → 6.2 ms | 11.6 → 6.5 ms | **61.0 → 31.3 ms** |
| XT32 100번, 유효 반환 126,196점 | 25.7 → 5.5 ms | 2.3 → 1.9 ms | **42.6 → 21.7 ms** |

두 실제 점군의 특징·인덱스와 동일한 모델 출력으로 만든 최종 박스가 변경 전후 비트 단위로 일치했습니다. 무작위·고밀도·범위 경계·40,000 pillar 초과 입력도 이전 NumPy 구현과 비교합니다. NumPy 1.26/2.2 호환성과 네이티브 버퍼 경계·동시 호출·패키지에서 추출한 라이브러리 실행도 검사합니다. 정렬 작업 공간은 호출 스레드당 약 0.88 MiB로 고정 재사용합니다.

```bash
~/.local/share/lidar_viewer/detector/runtime/bin/python tools/benchmark_detector.py \
  --model-dir ~/.local/share/lidar_viewer/detector/centerpoint-tiny-v4.1 \
  --cloud dataset/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/0000000041.bin \
  --iterations 100 --output /tmp/lidar-detector-benchmark.json
```

실제 KITTI 스캔과 XT32 bag에서 추론·박스 출력까지 확인했지만 **이 XT32 환경의 정답 라벨 기반 AP/recall을 측정한 것은 아닙니다.** 모델의 학습 센서와 장착 위치가 달라 오검출·미검출이 생길 수 있습니다. 현재는 단일 스캔 검출이며 시간 추적 ID나 속도 추정은 제공하지 않습니다. 다른 클래스나 현장 정확도 향상에는 해당 센서 데이터로 평가·추가 학습이 필요합니다.

원본 encoder → 독립 NCHW scatter → 원본 head와 합친 ONNX의 수치 일치, 특징·좌표·NMS 및 프로세스 종료/파이프/최신 프레임 전달 검사는 다음으로 확인합니다. 가중치 경로를 생략한 Python 검사는 NumPy만 필요합니다.

```bash
ctest --test-dir build --output-on-failure
python3 tests/test_centerpoint.py
~/.local/share/lidar_viewer/detector/runtime/bin/python tests/test_centerpoint.py \
  --model-dir ~/.local/share/lidar_viewer/detector/centerpoint-tiny-v4.1 \
  --cloud dataset/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/0000000041.bin
./build/lidar_viewer /path/to/scan.bin --detect --inspect
```

출처와 라이선스는 `third_party/licenses/Autoware-CenterPoint-NOTICE.txt`와 모델 카드에 기록합니다.

## 앱 아이콘

점군·3D ROI 박스·스캔 링을 표현한 `resources/icons/lidar-studio.svg`가 아이콘 원본입니다. Windows 실행 파일에는 다중 해상도 ICO, macOS 앱 번들에는 ICNS, Ubuntu에는 hicolor PNG/SVG 및 desktop 연결을 포함합니다. 체크아웃에서 직접 실행할 때에도 창/작업 표시줄 아이콘을 지정합니다. 생성한 파일을 저장소에 포함하므로 일반 빌드에는 이미지 변환 도구가 필요 없습니다. 아이콘을 수정한 후에는 `uv run --with pillow --with cairosvg tools/generate_app_icons.py`로 다시 생성할 수 있습니다.

소스의 `build/lidar_viewer`를 직접 실행할 때 Ubuntu 독·앱 메뉴의 아이콘을 연결하려면 한 번 등록합니다. 창 아이콘과 별도로 GNOME이 사용하는 `lidar-viewer.desktop` 항목이 필요합니다.

```bash
cmake --build build --target install-user-desktop
# 다른 빌드 폴더를 사용할 때
python3 tools/install_linux_desktop.py --binary /absolute/path/to/lidar_viewer
```

등록은 sudo 없이 사용자 `~/.local/bin/lidar-viewer`와 `~/.local/share/applications/lidar-viewer.desktop`, 사용자 아이콘 폴더에만 기록합니다. `XDG_DATA_HOME`도 따릅니다. 이후 앱 메뉴의 **LiDAR Studio** 또는 기존 `build/lidar_viewer` 실행 모두 같은 아이콘에 연결됩니다. 이미 실행 중인 창은 앱을 다시 켜면 새 등록을 사용합니다. 배포 패키지는 시스템 desktop 항목을 함께 설치합니다.

## FugSeg 지면분리

**Ground → Method → FugSeg**에서 [FugSeg 원본 구현](https://github.com/Leo-YuLi/FugSeg)을 선택할 수 있습니다. [IEEE TITS 2026 논문](https://doi.org/10.1109/TITS.2026.3682176)의 불확실성 기반 극좌표 격자 방식으로, 반사 잡음·가림 구간을 고려해 지면 높이를 전파한 뒤 점 단위로 분리합니다. CPU에서 실행하며 별도 학습 모델이나 GPU는 필요 없습니다. 기존 방식을 포함해 총 6종을 선택할 수 있습니다.

원본 commit `623f860cce70ffe0e3f673cd246254938ce8babe`의 코어를 고정하고, ROS 1·데이터셋 파일 I/O를 빌드에서 제외했습니다. 센서 연동은 `cmake/FugSeg.cmake`의 작은 생성 헤더 API로 처리합니다. 원본 분할 소스는 수정하지 않으며 다음을 보완합니다.

- Align에서 계산한 실제 센서 높이와 Ground reference를 사용하고, KITTI 차량에 고정된 차체 제외 영역을 해제합니다.
- 원본의 134,400점 고정 버퍼를 입력 크기에 맞춰 확장합니다. NaN/Inf·범위 밖 점도 장애물 쪽에 보존합니다.
- 원본 기본값(120개 각도 구간, 1m radial bin, 0.15m 두께, 7도 기울기 변화)을 제공하고 프로파일에 저장합니다. 고급 설정의 센서 잡음 기본값은 원본 KITTI 값이므로 현장 측정값에 맞게 조정할 수 있습니다.

정답이 정해진 평지·약 7도 경사와 45cm 이상 높이의 장애물 검사에서 FugSeg의 precision/recall은 모두 1.0이었습니다. 15만 점 이상 입력, 가까운 지면, 센서 높이 0.3/0.57/1.73m 변경, 지면 기준 Z 이동과 전체 점 보존도 검사합니다. 이 제한된 회귀 검사는 실제 XT32 환경 전체의 정확도 점수가 아닙니다.

현재 PC에서 XT32 실제 bag 2개·6개 프레임(각 128,000점)의 FugSeg CPU 처리 시간은 약 3.8–9.0ms였습니다. 센서 변환 뒤 지면분리 함수의 시간이며 로딩·렌더링을 포함하지 않습니다. 프레임별 지면 점 수는 정답 라벨 없이 정확도로 해석하지 않습니다.
