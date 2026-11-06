# LiDAR Studio

LiDAR 점군을 확인하고 지면 분할, IMU 수평 보정, ROI 편집, 객체 검출·추적을 수행하는 데스크톱 앱입니다.

## 설치

[Releases](https://github.com/YooJuno/lidar_studio/releases)에서 배포본을, [Actions](https://github.com/YooJuno/lidar_studio/actions)에서 최신 main 빌드의 아티팩트를 받습니다.

| 운영체제 | 실행 | 지원 입력 |
| --- | --- | --- |
| Ubuntu 24.04 | `.deb` 설치 후 `lidar_studio` | BIN, ROS 2 bag, 실시간 ROS 2 |
| Windows | ZIP을 풀고 `lidar_studio.exe` | BIN |
| macOS | `lidar_studio.app` 실행 | BIN |

Ubuntu의 bag·실시간 기능은 ROS 2 Jazzy가 필요합니다. `.deb` 설치 명령은 `sudo apt install ./lidar_studio-*.deb`이며 패키지 ID는 `lidar-studio`입니다.

## 소스 빌드

Ubuntu 기준입니다. ROS 2 bag·실시간 기능을 사용하려면 빌드 전에 Jazzy 환경을 불러옵니다.

```bash
sudo apt install cmake g++ libglfw3-dev libglew-dev libpcl-dev libnanoflann-dev zenity
git clone --recursive https://github.com/YooJuno/lidar_studio.git
cd lidar_studio
# ROS 2 기능을 사용할 때: source /opt/ros/jazzy/setup.bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/lidar_studio
```

[Windows·macOS 빌드 및 고급 설정](https://github.com/YooJuno/lidar_studio/blob/main/docs/guide.md#플랫폼별-빌드배포)

## 사용

입력 없이 실행한 뒤 **Open file**에서 `.bin`, `.npy`, `.pcd`, `.mcap`, `.db3` 또는 bag 폴더의 `metadata.yaml`을 선택합니다. 파일을 창에 드롭하거나 명령행으로 열 수도 있습니다.

```bash
./build/lidar_studio /path/to/scan.bin
./build/lidar_studio /path/to/bag --topic /b2/points
```

| 메뉴 | 기능 |
| --- | --- |
| **Live → Connect** | 실행 중인 ROS 2 센서 구독, 화면 정지·재개, 원본 bag 녹화 |
| **Align** | 센서 높이·회전 설정, IMU 자동 수평 보정 |
| **Ground** | 지면 분할 방식 선택, 지면만·장애물만 표시 |
| **ROI → Display ROI** | 체크한 박스 안의 포인트만 표시 |
| **ROI → Ground exclusion** | 박스 안의 점을 지면 검출 입력에서 제외. 편집 후 **Apply** |
| **Objects** | 사람·차량 등 객체 검출, 클러스터 표시, 객체 ID 추적 |
| **Datasets → Data** | ROI 추출 폴더·NPY·PCD를 열고 목록·방향키로 샘플 확인 |
| **Train** | LitePT·CenterPoint 학습 환경 검사, 파라미터 설정, 학습·중지·로그 확인 |

센서 드라이버는 별도로 실행합니다. 학습 모델 검출은 [모델 설치](https://github.com/YooJuno/lidar_studio/blob/main/docs/guide.md#사람차량-인식--원본-tier-iv-모델)가 필요합니다.

**Space** 재생·정지 · **←/→** 프레임 이동 · **R** ROI 그리기 · **Ctrl+O** 파일 열기 · **Ctrl+S** 프로필 저장

## 학습·라벨링용 내보내기

**ROI → Display ROI → Export points…**에서 현재 프레임 또는 프레임 구간·추출 간격·좌표계·저장 폴더를 선택합니다. 활성화한 ROI들의 합집합을 저장하며 진행률 확인과 취소를 지원합니다.

- `points/*.npy`: 원본 intensity를 보존한 `float32`, `N×4` 배열 `[x, y, z, intensity]`
- `pointcloud/*.pcd`: 라벨링용 동일 점군
- `metadata.json`, `metadata/*.json`: 데이터셋 목록, 원본 프레임·시각·ROI·IMU·좌표 변환

`20261028_134010_f14797_roi1-2.npy`처럼 기록 시각·프레임·ROI로 짧게 저장합니다. 전체 원본명·박스 좌표·좌표계는 JSON에 보존합니다. 다시 내보내면 번호를 붙여 기존 파일을 보존합니다.

내보낸 점군은 **미라벨링 데이터**입니다. 객체 박스·클래스 또는 점별 지면 라벨은 별도로 작성합니다.

## 데이터셋·학습

**Datasets**에서 내보낸 데이터셋 폴더를 엽니다. **←/→**로 넘기며 저장된 좌표 그대로 확인하고, **Train**에서 모델·라벨링한 데이터·출력 폴더를 선택합니다. 학습 중에도 뷰어를 사용할 수 있습니다.

- **LitePT**: 기존 `training.run` 학습 프로젝트와 CUDA Python 환경을 연결합니다. 점별 배경·열차 전면부·철제 라벨이 필요합니다.
- **CenterPoint**: MMDetection3D 학습 환경과 3D 박스 라벨이 필요합니다. Linux CUDA 환경 설치: `python3 tools/setup_centerpoint_training.py --python python3.11`.

**Check setup**으로 환경·라벨·학습/검증 분리를 검사한 뒤 **Start training**을 누릅니다. [라벨 형식·모델별 설정](https://github.com/YooJuno/lidar_studio/blob/main/docs/guide.md#데이터셋-열람과-학습)

## 개발·배포

```bash
ctest --test-dir build --output-on-failure
```

GitHub Actions에서 Ubuntu·Windows·macOS를 빌드·검사하고 패키지를 생성합니다. `v*` 태그는 검사 통과 후 GitHub Release에 배포합니다.

[상세 사용법·센서 설정·알고리즘·검증 기록](https://github.com/YooJuno/lidar_studio/blob/main/docs/guide.md)
