# 설치, 실행 순서, 안전과 정지 순서

## 1. 설치

**메인 PC** (Ubuntu 22.04, NVIDIA GPU, ROS 2 Humble)
1. [effmass-mpc](https://github.com/ukhyeon/effmass-mpc) 를 `~/effmass_mpc` 에 받고, 그 README 의 실행 환경(Isaac Sim 4.5 의 python, NVIDIA GPU MPC 라이브러리, NVIDIA Warp)을 갖춘다. MPC 는 Isaac Sim 의 python 으로 돌지만 시뮬레이터 화면은 쓰지 않는다.
2. 이 저장소의 `main_pc/` 를 같은 경로로 복사한다: `rsync -a main_pc/ ~/effmass_mpc/`.
3. `~/effmass_mpc/ws` 에서 `effmass_msgs`(사람 데이터 메시지), `effmass_robot`(FR3 로봇 구별 역관성)을 `colcon build` 한다.
4. FR3 v2.1 의 로봇 설정(franka_description 2.9.0 의 xacro 로 만든 URDF, 충돌 구 목록)을 `goal1/fr3v2_1/` 에 만든다. 타사 자산에서 나온 것이라 넣지 않았다.
5. `goal1/config/hosts.env.example` 을 `goal1/config/hosts.env` 로 복사해 채운다. 이 파일은 저장소에 넣지 않는다.
   - `CONTROL_PC`: 제어 PC 로그인(사용자@주소).
   - `FASTDDS_MAIN_PC_XML`: `goal1/config/fastdds_main_pc.xml.example` 에 주소를 채운 사본.
   - `HUMAN_PERCEPTION_START`, `HUMAN_PERCEPTION_PROC`, `HRC_WS`: 사람 인식(비공개 저장소 hrc_ws)을 켜는 명령, 그 프로세스를 찾는 문자열, 작업공간 위치.
6. `goal1/config/h1_choice.env.example` 을 `goal1/config/h1_choice.env` 로 복사한다. 실행 설정(MPS 몫, 프로파일, MPC 옵션, 질량 모드)이며, 예시는 2026-11-24 실행 설정이다.
7. 사람 데이터: [human_interface.md](human_interface.md) 의 형식으로 넣는다. 사람 인식과 사람 노드(`effmass_human`)는 hrc_ws 쪽 코드라 이 저장소에 없다.
8. 다른 위치에 설치했으면 `EFFMASS_ROOT` 에 effmass-mpc 폴더를 준다(기본 `~/effmass_mpc`).

**제어 PC** (실시간 커널, Docker)
1. Franka 의 [franka_ros2](https://github.com/frankaemika/franka_ros2) 를 Docker 로 설치하고, PREEMPT_RT 커널과 FCI 사용 설정을 Franka 문서대로 한다. 로봇 주소는 franka_bringup 설정 파일에 적는다(저장소에 넣지 않음).
2. 브리지(ros2_control 컨트롤러, Ruckig)와 게이트웨이를 franka_ros2 작업공간에서 컨테이너 안에 빌드한다. `real_run.sh` 는 제어 PC 의 `~/franka_ws/controller_dev/real_step3/run_real_step3.sh` 를 tmux 세션 "franka" 에서 부른다.

## 2. 실행 순서

1. **안전 점검**: 비상정지 버튼을 손 닿는 곳에 둔다. Franka Desk 에서 관절 잠금을 풀고 FCI 를 켠다. 작업 공간에 사람 외의 물체가 없는지 본다.
2. **사람 쪽 켜기** (메인 PC): `goal1/real_prep5/rp5_human_up.sh start`. MPS, 사람 인식, 사람 데이터 발행을 순서대로 켜고, 사람 데이터가 15 Hz 이상 오는지 확인한다. `rp5_human_up.sh status` 로 상태를 본다.
   - 카메라 없이 시험: 합성 사람(`rp5_synth_human.py`) 또는 기록 재생(`rp5_replay_human.py`)을 쓴다([human_interface.md](human_interface.md)).
3. **실행** (메인 PC): `goal1/scripts/real_run.sh <장면> <MPC 초> --human [--smooth]`.
   - 제어 PC 에서 홈 자세(⓪)를 만들고, 브리지·게이트웨이(⑤)를 켠 뒤, 사람 데이터 빈도와 GPU 를 확인하고 MPC 를 돌린다. 끝나면 정지 순서까지 마치고 결과를 정리한다.
   - 이 저장소에는 장면 `c1k`(`goal1/real_prep5/demo/scene_C1k_rp5floor.json`)만 있다. 다른 옵션은 `real_run.sh` 머리 설명에 있다.
   - 사람 기록 재생: `--human-replay <이름>_human.npz` 로 이전 실행의 사람 데이터를 같은 시각에 다시 넣는다(`rp5_replay_human.py`).
4. **끄기** (메인 PC): `goal1/real_prep5/rp5_human_up.sh stop`. MPC 나 `real_run.sh` 가 도는 동안에는 거부한다. 사람 데이터 발행, 사람 인식, MPS 순서로 끈다.

## 3. 안전과 정지 순서

- 실행 중에는 비상정지 버튼을 누를 수 있는 사람이 로봇 옆에 있다.
- 제어 PC 쪽 실행 중에는 ROS 명령줄 도구(`topic hz`, `topic echo`, `param get`, `control list_controllers`)를 쓰지 않는다. 실시간 루프와 겹쳐 로봇이 `communication_constraints_violation` 으로 멈춘 일이 있다. 기록은 ros2 bag 만 쓴다.
- 브리지는 단계 시작 전에 한 번만 켜고, 실행 중에 껐다 켜지 않는다.
- **정지 순서** (`run_real_step3.sh`):
  1. 목표 보내기를 멈춘다. 브리지는 목표가 0.1 s 끊기면 가속도·저크 한계 안에서 감속해 그 자리에 선다.
  2. 명령을 쥔 컨트롤러(브리지)를 `switch_controllers --deactivate` 로 끈다. 로봇의 제어 모드가 정상 종료된다.
  3. 컨트롤러가 꺼졌는지와 launch 로그의 새 오류(reflex, libfranka)를 확인한 뒤 launch 를 내린다. 게이트웨이는 그 뒤에 내린다.
- 메인 PC 에서 Ctrl+C 는 MPC 프로세스에만 SIGINT 를 보낸다. 제어 PC 쪽 정지 순서는 위와 같이 그대로 진행된다.
- 관절 상태가 이상하면 브리지가 멈춘다(`bad_state_stop_cycles`).
- 사람 근처 로봇 구 속도 상한은 설계값이며 표준 준수 주장이 아니다.
