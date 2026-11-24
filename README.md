> **English summary.** Real-robot side of [effmass-mpc](https://github.com/ukhyeon/effmass-mpc): a real-time human-avoidance MPC (GPU MPPI, NVIDIA GPU MPC library) on a Franka FR3 whose clearance zone around each body part is set from the human's directional effective mass.
> The MPC runs on a main PC at 60 Hz and sends joint targets over a wired LAN to a control PC (ROS 2 gateway, ros2_control bridge with Ruckig, 1 kHz, FCI).
> A comparison with uniform avoidance is in preparation.

# effmass-mpc-franka: 실물 Franka FR3의 유효질량 사람 회피 MPC

effmass-mpc-franka는 사람의 방향별 유효질량으로 이격 구역을 정하는 실시간 사람 회피 MPC를 실물 Franka FR3에서 실행하는 구성이다.

[시뮬 저장소: effmass-mpc](https://github.com/ukhyeon/effmass-mpc) · [석사 연구: human-robot-safety-system](https://github.com/ukhyeon/human-robot-safety-system) · [이전 프로젝트: franka-depth-mppi](https://github.com/ukhyeon/franka-depth-mppi)

![실물 실행: 왼쪽 실제 카메라 영상과 SMPL, 오른쪽 같은 순간의 3D 재구성](media/demo_effmass_fr3.gif)

*유효질량 회피로 실물 FR3를 실행한 장면이다(30 s 실행 중 약 5 s).*

화면 읽는 법:
- **왼쪽**: 실제 RGB 카메라 영상 위에 사람 인식이 추정한 SMPL 몸(회색)을 겹쳐 그렸다.
- **오른쪽**: 같은 순간의 3D 재구성이다. 로봇 관절은 기록된 측정값이고 사람은 MPC가 받은 사람 데이터다.
  - **주황 볼륨**: 팔 부위마다의 방향별 이격 구역이다. 유효질량이 큰 방향일수록 두껍다.
  - **빨간 선**: 그 순간 MPC가 계획한 앞으로 0.5 s의 손끝(TCP) 경로다.
  - **글자 상자**: 로봇에 가장 가까운 사람 부위, 그 방향의 유효질량, 거리다.

로봇이 사람 몸의 어느 부위에 어느 방향으로 다가가는지에 따라, 그 방향으로 사람 몸이 받아내는 유효질량을 매 제어 주기마다 계산해 이격 구역의 두께를 정한다. 방법과 시뮬레이션 비교는 [effmass-mpc](https://github.com/ukhyeon/effmass-mpc)에 있다.

이 저장소에는 실물 구성의 메인 PC 코드가 들어 있다.

- **실시간 MPC**: 60 Hz로 실행되는 MPC 실행기와 유효질량 비용 커널
- **사람 입력 연결**: 사람 데이터 수신, 급변 거르기, 로봇 받침 위 사람 후보 제외, 어깨 구 추가
- **실행·분석·시각화 스크립트**: 실물 실행(`real_run.sh`), 거리 분석(`exp_ana.py`), 위 데모 형식의 영상(`run_v5.sh`)
- **카메라 없는 실행**: 합성 사람과 이전 실행의 사람 기록 재생

## 1. 시스템 구성

![시스템 구성: RGB-D 카메라와 사람 인식, 사람 노드, MPC(메인 PC) -> 게이트웨이, ros2_control 브리지(제어 PC) -> Franka FR3](media/system_architecture.png)

- **메인 PC**: 사람 인식과 MPC가 GPU 하나를 나눠 쓴다(NVIDIA MPS, 사람 인식 쪽 SM 50 %). MPC는 시뮬레이터 화면 없이 NVIDIA GPU MPC 라이브러리로 실행된다. 사람 데이터와 관절 상태는 각각 별도 프로세스가 받아 공유 메모리에 최신 1개만 둔다.
- **사람 인식**: 비공개 저장소 hrc_ws이며 이 저장소에는 넣지 않았다. RGB-D 영상에서 SMPL 몸을 추정하고 사람 노드가 분절 19개의 질량중심·속도·역관성과 부위별 형상 24개(구·캡슐)를 ROS 2 메시지로 낸다. MPC가 받는 형식(입력 인터페이스)은 [docs/human_interface.md](docs/human_interface.md)에 있다. 이 형식을 따르면 다른 인식으로 바꿀 수 있고 합성 사람으로 카메라 없이 실행할 수도 있다.
- **제어 PC**: 게이트웨이가 유선 LAN 쪽 ROS 네트워크와 브리지 쪽 localhost 네트워크를 연결한다. 브리지(ros2_control 컨트롤러)는 60 Hz 목표를 Ruckig로 1 kHz 명령으로 보간한다. 목표가 0.1 s 끊기면 가속도·저크 한계 안에서 감속해 멈춘다.

**실시간 기록** (데모 실행 1개, 30 s, 1800 스텝; 실행 요약 json의 값):
- MPC 발행 60.0 Hz.
- 스텝 계산 시간: 중앙값 11.1 ms, 99 % 16.8 ms. 계산이 1/60 s를 넘은 스텝은 20개(1.1 %)다.
- 정해진 시각보다 늦게 발행한 스텝은 67개(3.7 %, 최대 22.8 ms)이고 건너뛴 스텝은 0개다.
- 받은 사람 데이터: 30 s 동안 605개(약 20 Hz).

## 2. 방법

사람의 부위(형상) 하나와 가장 가까운 로봇 구 사이에서, 로봇 구에서 그 부위를 향하는 단위 방향 u를 잡는다. 그 방향의 사람 유효질량과 이격 구역 두께는 다음과 같다.

```
m_h = 1 / (u^T Λ_h^{-1} u)                          Λ_h^{-1}: 그 부위가 속한 분절의 작업공간 역관성 (사람 데이터)
eta = clamp( d0 + k * max( sqrt(m_h / m_iso) - 1, 0 ),  d0,  eta_max )
```

- m_iso는 부위별 ISO/TS 15066 기준 질량이다. 유효질량이 기준보다 큰 방향만 두꺼워진다.
- 비용은 이격 구역(eta 안으로 들어가지 않게)과 그 바깥의 위험 필드(경계 전부터 미리 돌아가게)로 이뤄진다.

| 기호·설정 | 옵션 | 값 (데모 실행, 2026-11-24) | 뜻 |
|---|---|---|---|
| 질량 모드 | `--human-mass-mode human` (`MASS_MODE=human`) | 사람 유효질량 m_h | 위 식 |
| d0 | `--human-d-base` | 0.10 m | 팔 부위 기본 두께. 균일 회피의 두께와 같다 |
| k | `real_run.sh --k-gain` | 0.10 | 질량 이득. 균일 회피는 k = 0 |
| eta_max | `real_run.sh --eta-max` | 0.40 m | 두께 상한 |
| 어깨 | `--human-shoulder 1` | 기준 질량 3 kg | 팔 부위에 포함 |
| 팔 밖 부위 | `real_run.sh --other-d-base 0.03 --other-k-gain 0` | d0 0.03 m | 질량 항 없음 |
| 로봇 link0·link1 | `--human-robot-exclude link0,link1` | 구 6개 | 사람 비용에서 제외 |
| 로봇 받침 위 사람 후보 | `--human-base-exclude goal1/config/base_exclude.yaml` | | 사람으로 쓰지 않음 |
| 가슴 방향 | `HUMAN_CHEST_SPINE3=1` | | SMPL spine3 회전 반영(사람 노드 쪽) |
| W | `--human-field 1 --hf-wt 100` | 100 | 위험 필드 높이 |
| 속도 | `real_run.sh --speed 0.50 --smooth` | 0.5 m/s | MPC 명령 속도 상한 |
| MPC 계획 경로 기록 | `--rec-plan 1` | 0.5 s 앞 | 데모 영상의 빨간 선 |

전체 실행 설정 예시: [main_pc/goal1/config/h1_choice.env.example](main_pc/goal1/config/h1_choice.env.example)

## 3. 요구 사항

하드웨어:
- 1x Franka Research 3 (FCI)
- 1x RGB-D 카메라 (Azure Kinect)
- 2x PC: 메인 PC(사람 인식과 MPC, NVIDIA GPU), 제어 PC(실시간 커널)
- 두 PC를 잇는 유선 LAN

메인 PC는 Ubuntu 22.04, ROS 2 Humble, RTX 2080 Ti에서 시험했다. 제어 PC는 PREEMPT_RT 커널과 Docker 위의 franka_ros2를 쓴다.

| 의존성 | 비고 |
|---|---|
| effmass-mpc | 공통 코드(`step2_scene.py`, `step3_ros_common.py`, `g1_human_link.py`). 이 저장소의 `main_pc/`를 그 위에 같은 경로로 복사한다 |
| Isaac Sim 4.5의 python, cuRobo (NVIDIA GPU MPC 라이브러리), NVIDIA Warp | 별도 설치. 시뮬레이터 화면은 쓰지 않는다 |
| FR3 로봇 설정 (franka_description의 FR3 v2.1 URDF, 충돌 구) | 타사 자산에서 나온 것이라 넣지 않았다. `goal1/fr3v2_1/`에 만든다 |
| ISO/TS 15066 부위별 값 | `goal1/data/iso_15066.json` (effmass-mpc와 같이 넣지 않았다) |
| hrc_ws (사람 인식, 사람 노드) | 비공개 저장소 |
| 제어 PC: franka_ros2, libfranka, Ruckig | 실시간 커널, Docker |

## 4. 설치와 실행

명령은 메인 PC의 `~/effmass_mpc`에서 실행한다.

1. effmass-mpc를 `~/effmass_mpc`에 받고 이 저장소의 `main_pc/`를 그 위에 복사한다. 다른 위치면 `EFFMASS_ROOT`를 준다.
   ```bash
   rsync -a main_pc/ ~/effmass_mpc/
   ```
2. 메시지와 로봇 패키지를 빌드한다.
   ```bash
   cd ~/effmass_mpc/ws && colcon build --packages-select effmass_msgs effmass_robot
   ```
3. 설정 예시 세 개를 복사해 채운다. `hosts.env`는 제어 PC 주소와 사람 인식 시작 명령, `fastdds_main_pc.xml`은 두 PC 주소, `h1_choice.env`는 실행 설정이다.
   ```bash
   cd goal1/config && cp hosts.env.example hosts.env && cp fastdds_main_pc.xml.example fastdds_main_pc.xml && cp h1_choice.env.example h1_choice.env
   ```
4. 사람 쪽을 켠다. MPS, 사람 인식, 사람 노드를 순서대로 켜고 사람 데이터가 15 Hz 이상 오면 정상이다.
   ```bash
   goal1/real_prep5/rp5_human_up.sh start
   ```
5. 실물을 실행한다. 제어 PC의 홈 자세와 브리지를 켜고 사람 데이터와 GPU를 확인한 뒤 MPC를 실행하고 정지 순서까지 마친다. 결과는 `goal1/real_prep4/<이름>_*`에 남는다.
   ```bash
   goal1/scripts/real_run.sh c1k 30 --human --speed 0.50 --smooth --method effmass --k-gain 0.10 --eta-max 0.40 --other-d-base 0.03 --other-k-gain 0
   ```
6. 사람 쪽을 끈다.
   ```bash
   goal1/real_prep5/rp5_human_up.sh stop
   ```
7. 분석과 그림을 만든다. `exp_ana.py`는 거리와 회피 시작 거리를, `run_v5.sh`는 위 데모 형식의 PNG / MP4 / GIF를 만든다. Isaac 재렌더와 SMPL 겹치기에는 hrc_ws의 SMPL 패키지와 몸 모델이 필요하다.
   ```bash
   python3 goal1/real_prep5/exp1121/exp_ana.py one <실행 이름>
   goal1/real_prep5/exp1121/viz2/v5/run_v5.sh <실행 이름>
   ```

카메라 없이 시험하려면 합성 사람(`rp5_synth_human.py`)이나 기록 재생(`real_run.sh --human-replay <이름>_human.npz`)을 쓴다. 자세한 설치, 실행, 안전과 정지 순서: [docs/setup_and_run.md](docs/setup_and_run.md)

## 5. 저장소 구조

```
main_pc/goal1/scripts/      MPC 실행기(rp2_mpc_rt.py), 유효질량 비용 커널(g1_meff_gpu.py), 실물 실행(real_run.sh), 사람 입력 거르기
main_pc/goal1/real_prep5/   사람 쪽 켜기·합성 사람, 기록 재생(h2cost/), 장면(demo/), 분석과 시각화(exp1121/)
main_pc/goal1/config/       프로파일, 받침 제외 영역, 설정 예시(hosts, FastDDS, 실행 설정)
main_pc/goal1/data/         어깨 구 형상, 부위별 기준 질량 덮어쓰기, ISO/TS 15066 표 형식 예시
main_pc/ws/src/             사람 데이터 메시지(effmass_msgs), FR3 로봇 구별 역관성(effmass_robot)
docs/                       사람 입력 인터페이스, 설치·실행·정지 순서
media/                      데모 GIF, 시스템 구성 그림
```

## 6. 진행 상황

균일 회피와의 실물 비교 영상은 준비 중이다.

## 참고문헌

1. ISO/TS 15066:2016, Robots and robotic devices - Collaborative robots.
2. G. Williams et al., "Information theoretic MPC for model-based reinforcement learning," ICRA, 2017.
3. B. Sundaralingam et al., "cuRobo: Parallelized collision-free minimum-jerk robot motion generation," arXiv:2310.17274, 2023.
4. L. Berscheid and T. Kröger, "Jerk-limited real-time trajectory generation with arbitrary target states," RSS, 2021. (Ruckig)
