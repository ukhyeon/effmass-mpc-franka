# 사람 데이터 인터페이스

MPC(`main_pc/goal1/scripts/rp2_mpc_rt.py --human shm`)는 사람을 **부위별 형상(구·캡슐) 24개와 분절 19개의 질량 정보**로 받는다. 사람 데이터를 아래 형식으로 넣으면 나머지(비용, MPC, 브리지, 로봇)를 그대로 재현할 수 있다. 카메라 없이 시험하려면 [합성 사람](#합성-사람으로-실행해-보기)을 쓴다.

```
사람 데이터 (이 형식)  --ROS 2 topic-->  rp5_human_relay.py  --공유 메모리-->  rp2_mpc_rt.py
                     /effmass/human_inertia_state                     (MPC, 스텝마다 최신 1개를 읽음)
                     effmass_msgs/HumanInertiaState
```

두 지점 중 하나에 데이터를 넣으면 된다.

| 넣는 지점 | 형식 | 쓰는 곳 |
|---|---|---|
| ROS 2 토픽 `/effmass/human_inertia_state` | `effmass_msgs/HumanInertiaState` ([main_pc/ws/src/effmass_msgs/msg/HumanInertiaState.msg](../main_pc/ws/src/effmass_msgs/msg/HumanInertiaState.msg)) | 실물 실행 (`real_run.sh --human`). relay 가 받아 공유 메모리에 쓴다 |
| POSIX 공유 메모리 (이름 기본 `rp5_human`, `real_run.sh` 는 `rr5_human`) | float64[509], 아래 표 | relay 없이 직접 쓸 때 (합성 사람 `rp5_synth_human.py`, 기록 재생 `rp5_replay_human.py` 가 이 방식) |

## 1. 전달 방식

- **ROS 2 토픽**: `/effmass/human_inertia_state`, QoS best effort, depth 5, volatile. ROS 도메인은 사람 데이터를 보내는 쪽의 도메인(기본 `ROS_DOMAIN_ID=30`, `ROS_LOCALHOST_ONLY=1`)이다. MPC 와 로봇 링크는 다른 도메인을 쓴다.
- **relay** (`main_pc/goal1/real_prep5/rp5_human_relay.py`): 토픽을 받아 공유 메모리의 레코드 하나를 덮어쓴다(대기열 없음, 최신 1개). 블록을 만들고 끝날 때 지운다.
- **MPC**: 60 Hz 스텝마다 레코드를 한 번 복사한다(seqlock). 새 레코드가 없으면 마지막 것을 계속 쓴다.

## 2. 공유 메모리 레코드 (float64[509], seqlock)

쓰는 쪽은 `seq` 를 홀수로 만든 뒤 쓰고, 다 쓰면 짝수로 만든다. 읽는 쪽은 `seq` 가 짝수이고 복사 전후가 같을 때만 받아들인다.

| 위치 | 길이 | 내용 |
|---|---|---|
| 0 | 1 | seq (홀수 = 쓰는 중) |
| 1, 2 | 2 | 입력 stamp (sec, nanosec): 사람 자세를 잰 카메라 영상의 시각 |
| 3 | 1 | 받은 시각 (`time.time()`, 시스템 시계 초) |
| 4 | 1 | 입력 순번 (보내는 쪽이 매기는 1부터의 번호) |
| 5 | 1 | 보내는 쪽 계산 시간 (ms, 기록용) |
| 6 | 1 | 받은 시각 (`time.monotonic()`) |
| 7 | 1 | 지금까지 쓴 레코드 수 |
| 8 | 57 | `com_position`: 분절 19개 질량중심 (19×3, m) |
| 65 | 57 | `com_velocity`: 분절 19개 질량중심 속도 (19×3, m/s) |
| 122 | 171 | `inv_inertia`: 분절별 3×3 작업공간 역관성 (19×9, 행 우선, 1/kg) |
| 293 | 72 | `shape_p0`: 형상 24개의 캡슐 한 끝 / 구 중심 (24×3, m) |
| 365 | 72 | `shape_p1`: 캡슐 다른 끝 / 구 중심 (24×3, m) |
| 437 | 24 | `shape_radius`: 반지름 (m) |
| 461 | 24 | `shape_segment`: 형상이 속한 분절 번호 (0..18) |
| 485 | 24 | `shape_type`: 0 = 구, 1 = 캡슐 |

크기가 다른 메시지(분절 19개, 형상 24개가 아닌 것)는 relay 가 버린다.

## 3. 모양 형식

**분절 19개** (순서 고정, 메시지의 `segment_names`):

| 번호 | 이름 | 번호 | 이름 | 번호 | 이름 |
|---|---|---|---|---|---|
| 0 | pelvis | 7 | right_foot | 14 | right_upper_arm |
| 1 | left_thigh | 8 | chest | 15 | left_lower_arm |
| 2 | right_thigh | 9 | neck | 16 | right_lower_arm |
| 3 | abdomen | 10 | left_shoulder | 17 | left_hand |
| 4 | left_calf | 11 | right_shoulder | 18 | right_hand |
| 5 | right_calf | 12 | head | | |
| 6 | left_foot | 13 | left_upper_arm | | |

**형상 24개**: 몸통 분절 3개(pelvis, abdomen, chest)는 형상 2개씩, 손 2개는 형상 2개씩(손바닥·손가락, 엄지), 나머지 분절 14개는 1개씩(머리는 캡슐). 각 형상은 `shape_segment` 로 분절에 묶인다. 비용은 형상마다 계산하고, 그 형상의 충돌 방향 유효질량은 소속 분절의 `inv_inertia` 로 구한다. 반지름이 0 인 형상은 쓰지 않는다.

**어깨 구** (`rp2_mpc_rt.py --human-shoulder 1`, 2026-11-24 실행 설정): MPC 가 빈 형상 자리 21, 22 에 왼쪽·오른쪽 어깨 구를 직접 쓴다. 중심은 각 어깨 분절의 질량중심(`com_position` 분절 10, 11), 반지름 0.082 m, 종류 구다(`goal1/data/shapes_v2_shoulder.json`). 사람 데이터 쪽에서 그 두 자리를 비워 두면 된다. 어깨의 ISO 기준 질량은 위팔 값 3 kg 을 쓴다(`goal1/data/iso_ref_override.json`).

**역관성** `inv_inertia` 는 분절 질량중심에서 본 작업공간 역관성 J Mᐨ¹ Jᵀ (3×3) 이다. 방향 u 의 사람 유효질량은 m_h = 1 / (uᵀ Λ⁻¹ u) 다. 역관성을 구할 수 없으면, 부위별 기준 질량(예: ISO/TS 15066 의 값)으로 Λ⁻¹ = I / m 을 넣어 방향과 무관한 질량으로 시험할 수 있다(이 경우 유효질량 회피는 방향 효과를 잃는다).

## 4. 좌표계

- 레코드의 위치와 속도는 **world 좌표계**(바닥 기준, 카메라 자세로 정한 좌표계)다.
- MPC 는 `--human-frame <yml>` 로 world → 로봇 base 변환을 받는다: `p_base = Rz(rotation_z_deg)ᵀ (p_world − base_position_world)`. 예: [main_pc/goal1/config/live_frame_rp5.yml](../main_pc/goal1/config/live_frame_rp5.yml) (회전 0, base 가 바닥에서 0.715 m 위).
- 처음부터 base 좌표로 넣으려면 `rotation_z_deg: 0`, `base_position_world: [0, 0, 0]` 인 파일을 쓴다.

## 5. 시각

- **입력 stamp** (위치 1, 2): 사람 자세를 만든 카메라 영상의 시각. MPC 는 매 스텝 `age = 지금(시스템 시계) − 입력 stamp` 를 구한다. 따라서 stamp 는 MPC 가 도는 PC 의 시스템 시계와 같은 시계여야 한다.
- 사람 위치 예측(`cv` 모드)을 켜면 각 형상을 소속 분절의 질량중심 속도로 `age + rollout 시각` 만큼 평행 이동한다. 기본은 현재 위치(`static`).
- **받은 시각** (위치 3, 6): relay 가 받은 시각. 사람 데이터 빈도와 지연 점검(`rp5_human_check.py`, `real_run.sh` 의 15 Hz 점검)에 쓴다.

## 6. 속도

- `com_velocity` 는 입력 stamp 차이로 구한 분절 속도다. 처음 메시지나 stamp 가 이상하면 0 을 넣는다.
- MPC 쪽 거르기(선택): `--human-jump-filter 1 --hjf-mode anchor` 는 몸통이 한 프레임에 크게 뛴 레코드를 버리고 마지막으로 받아들인 레코드를 쓴다. `--human-vel-clean 1` 은 버린 레코드의 속도를 0 으로, `--hvc-max`(기본 2.0 m/s) 를 넘는 분절 속도를 0 으로 둔다.

## 합성 사람으로 실행해 보기

`main_pc/goal1/real_prep5/rp5_synth_human.py` 는 카메라 없이 공유 메모리 레코드를 직접 쓴다(같은 형식, `--rate` Hz). 사람 한 명의 상태 하나를 강체로 옮겨 정면·옆·흔들기 동작을 만든다. 입력(`--cap`)은 위 형식의 상태를 담은 npz 다(키 `com_position` (N, 57), `com_velocity` (N, 57), `inv_inertia` (N, 171), `shape_p0`, `shape_p1` (N, 72), `shape_radius`, `shape_segment`, `shape_type` (N, 24); N 프레임 중 하나를 쓴다). 실제 사람에게서 잰 상태 파일은 개인 데이터라 넣지 않았다. relay 가 쓴 공유 메모리 레코드를 한 번 저장하거나(실물 실행의 `--rec-human 1` 기록에도 같은 필드가 있다), 위 형식으로 직접 만든 상태를 쓴다.

```
python3 main_pc/goal1/real_prep5/rp5_synth_human.py --cap <state.npz> --shm-name rp5_human --rate 30    # 동작·시각 옵션은 파일 머리 설명 참고
# 다른 터미널: MPC 를 --human shm --human-shm rp5_human 로 실행 (가짜 로봇: rp4_run.sh / real_run.sh --fake)
```

## 기록 재생

실물 실행에서 `rp2_mpc_rt.py --rec-human 1` 로 남긴 사람 기록(`<이름>_human.npz`)은 `main_pc/goal1/real_prep5/h2cost/rp5_replay_human.py --replay <파일> --shm-name <이름> --start-log <MPC 로그>` 로 같은 시각에 다시 공유 메모리에 쓸 수 있다. 같은 사람 동작을 두 방식에 넣는 짝 비교에 쓴다.
