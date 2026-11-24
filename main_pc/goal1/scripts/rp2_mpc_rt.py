"""rp2_mpc_rt.py - real_prep2: GPU MPC library MPC without Isaac (no SimulationApp), wall-clock 60 Hz, joint state in, JointTrajectory out.
Run with Isaac's python and the ROS paths: env_check/scripts/run_isaac_ros.sh (ROS_DOMAIN_ID 30, ROS_LOCALHOST_ONLY 1); see rp2_run.sh.
  input : sensor_msgs/JointState on --js-topic (default /NS_1/franka/joint_states), joints fr3_joint1..7 (matched by name); the latest
          message is the MPC's current state (position, velocity; acceleration 0)
  output: trajectory_msgs/JointTrajectory on /mpc/joint_trajectory every step, joint names fr3_joint1..7, header.stamp 0, K points of the
          horizon (positions, velocities, accelerations, time_from_start = horizon time; as step2b --jt-out). Infeasible step: one point
          at the measured position, zero velocity / acceleration (hold).
  profile: a step2b profile yml (default goal1/config/real_acc10.yml): robot_cfg (acceleration limit), pose_rot_w, sphere_speed_limit.
          Human-related keys (speed_cap*, meff_*, human_*) need the human cost channel, which is not built here (no human); they are
          listed in the log as not applied. sphere_speed_limit > 0: the published first point is moved toward the measured position so
          that no robot sphere moves faster than the limit in one step (as step2b --sphere-speed-limit; its velocity / acceleration
          are scaled by the same factor; points 2..K are not changed).
  scene : no human. Goals alternate between two hand poses (the end points of a scene json, default the live3m corrected path of
          motion 1, gripper down); a goal is switched on arrival (1 cm, 0.05 rad) or after LEG_MAX_STEPS. world = robot table + floor.
          HUMAN_HOOK below is the slot for a human input later (not used).
  timing: step k is scheduled at T0 + k / 60 s (wall clock, time.perf_counter); the loop sleeps until then, a late step starts at
          once (no step is skipped). overrun = the step's own computation (start to publish) > 1/60 s; deadline miss = publish later
          than schedule + 1/60 s.
options for the joint-state input (default off = as before):
  --js-source shm --shm-name rp2_js : read the latest joint state from shared memory written by rp2_js_relay.py (separate process),
          once at the start of each step; no ROS subscription in this process
  --js-qos best_effort : subscribe with the sensor-data QoS (best effort, depth 5) instead of reliable depth 1
  --age-comp 1 : use q + qd * age, age = now - header stamp of the state used (same clock on one computer)
  every step records the age by header stamp (stamp_age) and by receive time (state_age)
acceleration clip (real_prep3; default off = as before):
  --acc-clip 1 [--acc-max 9.5] [--vel-frac 0.9] : last stage before publishing (after the speed limiter): with the two targets sent
          last (p1, p2), v_prev = (p1 - p2) / dt; the first point p is moved so that per joint the differenced acceleration
          ((p - p1) / dt - v_prev) / dt is within +-acc_max and the velocity (p - p1) / dt within +-vel_frac x URDF velocity
          (dt = 1/60 s). The point's velocity becomes (p' - p1) / dt and its acceleration (v' - v_prev) / dt; points 2..K are not
          changed. Before the first step p1 = p2 = the measured position. --clip-feedback 1 (default with --acc-clip 1): the sent
          first point (position, velocity, acceleration) is written into GPU MPC library's state filter (its "previous command"), so the
          next start state blends the measurement with what was actually sent, not with the unclipped command.
start after warm-up (real_prep3, default on):
  --warm-reset 1 : after the warm-up solves and just before the first real step, GPU MPC library's state filter (its "previous command") is
          set to the latest measured joint state with zero velocity and acceleration. Without it the warm-up solves advance the filter
          and the first target is tens of mrad away from a resting robot. 0 = as before.
two-PC link (real_prep4, 2026-11-19; default off = as before):
  --link c6 : joint state from the control-PC gateway (default topic /c6/joint_states, joints matched by name: the gateway sends
          fr3_joint1,3,2,4..7), subscription best effort depth 1; JointTrajectory output best effort depth 1, header.stamp = send
          time (system clock), first point time_from_start 0 (points 2..K: horizon time minus the first point's), hold point 0 -
          the same as c6_t2/for_main_pc/traj_sender.py. Header stamps are on the control-PC clock: stamp_age is not meaningful.
  --stale-ms 50 : if the newest joint state was received (main-PC monotonic clock) more than this long ago, at the start of a step
          or just before publishing, nothing more is published and the run ends (no restart). Reason, step, wall time and age go
          to the log and to summary "stop". 0 = off.
  With --js-source shm the receive time comes from rp2_js_relay.py (record index 17, monotonic clock).
  SIGINT (Ctrl+C) during the loop (2026-11-20): nothing more is published from the next check (step start or just before
  publishing), the records are saved, summary "stop" reason "SIGINT". Before the loop SIGINT ends the process as before.
human input (real_prep5 H1, 2026-11-20; default --human off = as before, nothing below is built or imported):
  --human shm --human-frame <live_frame yml> [--human-shm rp5_human] [--wait-human 120] [--human-near 1]
          The human input of the 11-18/19 Isaac live runs (lv_live_kinect.sh: step2b --human-source inertia --human-frame ... --profile
          live_t1.yml --meff on --meff-mode gain ...), without Isaac: /effmass/human_inertia_state is received by a separate process
          (goal1/real_prep5/rp5_human_relay.py, its own ROS domain, e.g. 30 localhost) and kept as the latest message in shared memory;
          this process subscribes to nothing human-related and publishes nothing human-related. Each step HUMAN_HOOK copies the latest
          record into g1_human_link.HumanInertiaLink (node None, no display; same latest-slot fields as its ROS callback) and calls its
          pump(): frame transform (to_isaac), velocity EMA, tracking-loss hold, MeffInputs.set_human / set_age, as in step2b. After the
          joint state is read, the robot inverse inertia at the measured joints goes to MeffInputs.set_robot (RobotInertiaNP), as step2b.
          Cost: g1_meff_gpu.MeffChannel hooked in before the first solve with the live-run values: impl warp, u closest (u-def nearest),
          mode gain_part (gain, mu_ref per part), agg max_all, w 1e5, d_base 0.065, mu_ref 0.53, k_gain 0.10, eta_max 0.25, hfront 0;
          from the --profile: speed_cap, speed_cap_v / _dist / _w, meff_motion, human_vel_alpha, human_loss_hold_s, human_loss_detect_s
          (then applied, no longer listed as not applied; missing keys take the step2b defaults).
          The loop starts only after the first human record (wait --wait-human s, else exit 3 before anything is published).
          --human-near 1 (default with --human shm): after publishing, the minimum gap robot sphere - human shape (MeffChannel.query_all,
          surfaces, measured joints) is recorded per step (one GPU sync); 0 = off.
          Steps npz gets h_new, h_age, h_ms (pickup + pump), r_ms (robot inertia), near_min, near_sph, near_shape, near_seg, near_ms; the
          summary gets "human".
H2 (2026-11-20; defaults off = as before): --skip-late 1 (see the option help; summary skipped_slots, skip_events,
  late_at_start_max_ms; per step skip_before, late_start), profile key settings.mppi_particles (MPPI num_particles, e.g. 200).
output smoothing (real_prep5 overnight filter, 2026-11-20; defaults off = as before):
  --out-lpf <kind> [--out-lpf-hz 12] [--out-lpf-n 9] [--out-lpf-deg 1] [--out-lpf-m 4] [--out-lpf-feedback 1]
          causal smoothing of the published first point (goal1/scripts/rp5_out_filter.py: ema, bw2 = 2nd-order Butterworth 60 Hz bilinear,
          cd2, sg = polynomial end-point fit over the last n targets, blend = mean of the last m plans' predictions for this step; chains
          like blend+bw2). Applied after the speed limiter and before the acceleration clip; held / infeasible steps feed the hold position.
          Reset to the measured joints just before the first step. Velocity / acceleration of the point = finite differences of the
          smoothed positions. Points 2..K are not changed. --out-lpf-feedback 1: the smoothed point is written into the solver state
          filter ("previous command"); with --acc-clip 1 --clip-feedback 1 that write already happens after the clip (final value).
          For no feedback at all use --out-lpf-feedback 0 --clip-feedback 0.
  --lim-ref cmd [--lim-v 0] [--lim-guard 3] : the sphere speed limiter scales the first point toward the previously published point instead of the
          measured joints (default meas = as before), with a guard toward the measured joints at guard x limit.
  --mpc-override <yml> : merged into the MPC task settings through override_particle_file (on top of the seed override), e.g.
          {mppi: {sample_mode: MEAN}}; the merged file is written to <out>_mpc_override.yml.
  --rec-goal-err 1 : record per step the ee position / rotation error to the current goal (goal_pe, goal_qe) and the unsmoothed first
          point (raw_q); the summary gets "out_filter" (15 Hz share of sent and raw target acceleration, error at arrivals).
human-state recording (H2 prep, 2026-11-20; default off = as before):
  --rec-human 1 (needs --human shm): writes <out>_human.npz for replaying the run (e.g. in Isaac Sim) with the same human:
          msg_* = every relay record the MPC picked up (one per step at most; relay index msg_relay_idx shows records the relay got but
          the MPC never saw): step, pickup wall time, input stamp, relay receive wall / monotonic time, input count, compute ms, com,
          vel, inv (19 x 3 x 3), p0, p1, center, radius, segment, type (node world frame, as received; frame_R / frame_t = node world ->
          robot base as used by the MPC). step_* = per MPC step: publish wall time ns, relay index and pumped number of the record in
          use, its age (s), new flag, Top-K mask / count (with --hsel), nearest active human shape per robot sphere (distance m, segment)
          and per robot link (link_min, link_seg) at the measured joints, from the --human-near query (after publishing; one extra
          small GPU->CPU transfer per step). Summary: human.rec (messages, relay records not seen, file size).
human jump filter (H2, 2026-11-20; default off = as before):
  --human-jump-filter 1 [--hjf-jump 0.2] [--hjf-back 0.2] [--hjf-hold 1.5] (with --human shm): a human record whose torso (pelvis,
          abdomen, chest COM) moved more than --hjf-jump m from the previous record starts a hold; held records are replaced by the
          last accepted record's data (with the new stamp / receive time: no tracking-loss hold, age as usual) until the torso is back
          within --hjf-back m of the last accepted torso, or more than --hjf-hold s after the last accepted record (then taken as is).
          goal1/scripts/rp5_jump_filter.py; offline replay goal1/real_prep5/filt/jf_replay.py. Summary human.jump_filter.
  --hjf-mode hold (default, the rule above) | anchor (2026-11-20 evening, opt-in; rp5_jump_filter.AnchorFilter): accept a record when
          its torso step from the previous record is <= --hjf-jump (default 0.3 in this mode) AND it is within --hjf-back m of the last
          ACCEPTED torso; or when the torso has moved <= --hjf-settle-step (0.1) m per record for --hjf-settle-n (3) records in a row and
          no torso COM is within --hjf-near (0.25) m of a robot collision sphere surface (measured joints; 0 = no near test); or more than
          --hjf-hold-max (3.0) s after the last accepted record. --hjf-hold is not used. Comparison: goal1/real_prep5/filt/jf_cmp.py.
human base exclusion (baseex, 2026-11-24; default off = as before):
  --human-base-exclude <yaml> (with --human shm; e.g. goal1/config/base_exclude.yaml, goal1/scripts/rp5_base_exclude.py): a human
          record whose pelvis COM (segment 0, node world) is inside the robot base / cart box of the file is not a person (the sensing
          took the robot as the body): it is handled as a jump-filter rejection (the last used record's data, velocity 0 with
          --human-vel-clean, counted as rejected by --human-gap-slow) and is NOT given to the jump filter (its reference stays on the
          person). With no earlier record the record is dropped (as no message). Recorded as msg_jf_rejected 1 + msg_bex_excluded 1
          (records dropped with no earlier record are not in the human npz). Summary human.base_exclude; one line at the end.
human shift and leg timeout (cam20, 2026-11-24; default off = as before):
  --human-shift dx,dy,dz (with --human shm): every human record is moved by this vector (node world frame, m) before anything uses it
          (base exclusion, jump filter, shoulder spheres, cost, record): COM, shape end points p0 / p1; velocities and inertia unchanged.
          For replays of a recording made with another camera translation: shift = t(new camera pose) - t(recording's) (real_run.sh
          --camera-pose / .h1_choice CAMERA_POSE). Summary human.shift; the recorded msg_* are the shifted records.
  --leg-timeout S: a leg (A -> B or B -> A) ends after round(S * 60) steps without arrival (default: step2_scene LEG_MAX_STEPS 400 =
          6.67 s). Summary leg_timeout_steps / leg_timeout_s.
arrival tolerance (arrtol, 2026-11-24; default = as before):
  --arrive-pos-tol M: position tolerance of the goal switch on arrival (default step2_scene POS_TOL_M 0.01 m; the orientation
          tolerance 0.05 rad is unchanged); also the arrival count and the non-via goals of --goal-cycle. Summary arrive_pos_tol.
human cost robot sphere exclusion (robex, 2026-11-24; default off = as before):
  --human-robot-exclude link0,link1 (with --human shm): robot collision spheres whose link name ends with _<link> (RobotInertiaNP
          sphere_link; fr3v2_1: link0 2 spheres, link1 4 spheres) are left out of the human cost only: clearance zone and risk field
          (MeffChannel kernel, human_exclude), speed-cap near test (they are never near), Top-K shape selection (--hsel) and the field
          record (hf_*). Self collision, scene collision and the near_min record (all spheres) are unchanged. Summary
          human.robot_exclude; one log line.
human velocity clean-up and speed-cost range (H2 cost, 2026-11-20 evening; default off = as before):
  --human-vel-clean 1 [--hvc-max 2.0] (with --human shm): in the human reader, a segment velocity above --hvc-max m/s is set to 0 (the
          human node gave 22 m/s for the ghost frames); a record the jump filter rejected gets velocity 0 for every segment (the held
          record's data otherwise kept its velocity, which froze the EMA and the constant-velocity shift moved the arm 16 m away); on the
          first accepted record after a rejection the velocity EMA (human_vel_alpha) restarts. Works with --hjf-mode hold or anchor, and
          without the jump filter (then only the clip). Summary human.vel_clean.
  --human-d-base D (2026-11-20 evening): base clearance of the human cost (d_base = eta0 = eta_min of the gain_part cost; default 0.065
          as lv_live_kinect.sh). The mass-dependent part (k_gain 0.10) and eta_max 0.25 are unchanged (eta_max is raised to D if below).
  --human-gap-slow 1 [--hgs-v 0.14] [--hgs-age 0.1] [--hgs-rate 2.0] (2026-11-20 evening; with --human shm): while the jump filter rejects
          records (the last record read was rejected) or the human data is older than --hgs-age s, the sphere speed cap of the limiter
          (--lim-v with --lim-ref cmd, else the profile sphere_speed_limit) goes down to --hgs-v m/s; when the data is normal again it
          goes back. The cap moves by at most --hgs-rate m/s per s (no sudden deceleration / acceleration: 0.50 -> 0.14 in 0.18 s at 2.0).
          Steps npz: hgs_cap, hgs_blind; summary human.gap_slow (rejected / old / blind / slowed seconds, lowest cap).
  --human-field 1 --hf-wt W (2026-11-20 evening, H2 field; default off): effective-mass risk field on top of the clearance cost
          (g1_meff_gpu.py kernels): with d = robot sphere surface - human shape surface and d_req(u) = eta(u) of the gain_part cost
          (d_base + k_gain x part excess), an outer hill W (1 - (d - d_req) / L)^2 for d_req <= d < d_req + L with L(u) = d_req(u), and
          W + the old wall inside d_req (continuous at d_req); summed as before (max over shapes). Per step at the measured joints
          (with --human-near): steps npz hf_val / hf_sph / hf_seg (largest field cost over the active shapes); summary human.field.
  --human-depth-margin D [--hdm-camera <camera_pose yml>] (H2 field; default 0 = off): pelvis, abdomen and chest shapes are swept
          +-D m along the ray camera -> shape midpoint (exact closest point on the swept shape, kernel, near query, Top-K). Camera
          position t (world) from --hdm-camera, else --rec-camera-pose, else goal1/config/camera_pose_rp5.yaml; world -> base by the
          human frame. MPPI seed: --mppi-seed N (existing; default 0 = fixed seed).
  --infeas-escape 1 [--esc-step R] (night 2026-11-21; default 0 = hold as before, nothing computed): GPU MPC library marks a step infeasible
          from its one-step command (bound + world + self collision); rp2 then holds at the measured joints. When the MEASURED state is
          inside the self-collision buffer (the human cost pushed the arm back toward its own base), no command leaves it and the robot
          stood still to the end of the run (replay w100dm_at115: 542 steps). With the option such a step publishes one joint step of R rad
          (default 0.004 = 0.24 rad/s) down the self-collision gradient (central differences), only if the world term stays 0 and every
          joint keeps 0.05 rad to its limit; no constraint is relaxed. Steps npz inf_term (bits: 1 bound, 2 world, 4 self, 8 escaped),
          inf_self (self value of the command / measured state); summary infeasible_terms.
  --human-k-gain K (night 2026-11-21; default: 0.10 as before): mass-dependent part of the gain_part human cost (0 = uniform d_base).
          (kgain 2026-11-23) with --human-method effmass K is used (was: rejected unless 0.10); with uniform only 0 is accepted.
  --human-eta-max E (kgain 2026-11-23; default 0.25 as before): upper bound of the clearance thickness (raised to d_base if below).
  --human-other-d-base D --human-other-k-gain K2 (kgain 2026-11-23; default: not given = the arm values): base clearance and
          mass-dependent part of the NON-arm segments (all but left/right upper arm, lower arm, hand: torso, head, neck, shoulders,
          thighs, calves, feet); the arm segments keep d_base / k_gain. Same rule eta = clamp(D + K2 x part excess, D, eta_max), the risk
          field follows the segment's eta (d_req .. 2 d_req). Both methods alike (uniform: arm k_gain 0; K2 as given, default 0).
          Written to the summary (human.part_cost) and the steps npz (human_k_gain, human_eta_max, human_other_d_base,
          human_other_k_gain, human_arm_parts; actual values also when the options are not given).
  --human-shoulder 1 [--human-shoulder-json J] [--human-iso-override-json I] (shoulder 2026-11-24; default 0 = as before, nothing
          imported or changed): (a) every human record read from the relay (live or lockstep replay) gets two spheres in its empty slots
          21 / 22 (goal1/data/shapes_v2_shoulder.json: centre = the message's left / right_shoulder segment COM com[10] / com[11], radius
          0.082 m, segment 10 / 11; scripts/human_shoulder.py), so they are also in the --rec-human file; (b) left / right_shoulder join the
          arm parts (arm d_base / k_gain instead of --human-other-*); (c) their ISO reference mass is 3.0 kg (upper arm) instead of 40 kg
          (goal1/data/iso_ref_override.json; iso_15066.json unchanged). The shoulder joints stay locked in the human node (always_lock_joints
          10, 11). Summary human.part_cost / steps npz: human_arm_parts (8), human_shoulder 1, human_iso_override, human_shoulder_shapes.
  --human-mass-mode reduced|human (massmode 2026-11-24; default reduced = as before): reference of the effective-mass clearance.
          reduced: eta = d0 + k max(sqrt(mu / mu_iso) - 1, 0), mu = 1 / (u' inv_h u + u' inv_r u) (human-robot reduced mass, kernel mode
          gain_part). human: eta = d0 + k max(sqrt(m_h / m_iso) - 1, 0), m_h = 1 / (u' inv_h u) (human effective mass along u; kernel mode
          gain_part_h); m_iso = ISO per segment (iso_15066.json, shoulders 3 kg with --human-shoulder 1). The risk field follows eta (no mass
          term of its own), so it changes with the mode. Uniform method (k_gain 0): eta = d0 in both modes. Summary human.part_cost /
          steps npz human_mass_mode (always written).
  --goal-cycle 1 [--via-tol M --via-rot-tol R] (demo, 2026-11-20 night; default 0 = A <-> B toggle as before): the goals_wxyz of the
          scene are visited in order 0, 1, .., n-1, 0, .. (start at goal 1, q_start = goal 0 as before; with 2 goals this is the toggle).
          Goals whose index is in the scene key "via" (list of int) switch at --via-tol m / --via-rot-tol rad (pass-through waypoints,
          default 0.03 / 0.10); the others at the arrival tolerance (1 cm, 0.05 rad). Summary goal_cycle (arrivals per goal, laps =
          arrivals at goal 0, lap times); steps npz goal_arr (1 = switched on arrival).
  --speed-cap-dist D: overrides the profile's speed_cap_dist (m; e.g. 0.5 = the speed cost only within 0.5 m of the human, as live_t1).
  --rec-camera-pose <yml> (with --rec-human): camera pose file of the sensing run (IMU-corrected copy), path and text kept in
          <out>_human.npz next to the scene and profile paths and texts.
--rec-plan 1 (recplan 2026-11-24; default 0 = as before, nothing recorded): the MPC plan of every step, the horizon joint positions the
          solve gives (MPPI mean actions integrated from the filtered start state, the same hq whose first point is published, BEFORE the
          speed limiter / acceleration clip / output filter change that first point) up to 0.5 s ahead (all points if the horizon is shorter):
          steps npz plan_q (steps x points x 7, float32; NaN when nothing was planned: infeasible hold, escape step) and plan_t (points,
          s after the step's measured state = cumulative traj_dt). Copied from the horizon already moved to the CPU each step (no new sync).
Outputs (--out <prefix>): <prefix>_steps.npz (per step timing, sent points, state age, feasible, limiter scale), <prefix>_summary.json.
"""
import argparse, json, os, sys, threading, time, types
sys.path.insert(0, os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/env_check/scripts"))
ap = argparse.ArgumentParser()
ap.add_argument("--profile", default="~/effmass_mpc/goal1/config/real_acc10.yml")
ap.add_argument("--js-topic", default=None, help="default /NS_1/franka/joint_states; with --link c6 /c6/joint_states")
ap.add_argument("--jt-topic", default="/mpc/joint_trajectory")
ap.add_argument("--k", type=int, default=1, help="horizon points per JointTrajectory")
ap.add_argument("--duration", type=float, default=180.0, help="measured loop length, s (wall clock)")
ap.add_argument("--scene-json", default="~/effmass_mpc/goal1/live/live3m/corrected/cal/m1b/scene_AB.json")
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--wait-js", type=float, default=60.0, help="s to wait for the first joint state")
ap.add_argument("--out", required=True)
ap.add_argument("--mppi-seed", type=int, default=0)
ap.add_argument("--js-source", default="topic", choices=["topic", "shm"])
ap.add_argument("--shm-name", default="rp2_js")
ap.add_argument("--age-comp", type=int, default=0)
ap.add_argument("--warm-reset", type=int, default=1, help="(real_prep3) 1: reset the solver state filter to the measured state (velocity 0) after the warm-up; 0 = off")
ap.add_argument("--acc-clip", type=int, default=0, help="(real_prep3) 1: clip the differenced acceleration / velocity of the published first point")
ap.add_argument("--acc-max", type=float, default=9.5, help="(real_prep3) rad/s^2, per joint")
ap.add_argument("--vel-frac", type=float, default=0.9, help="(real_prep3) velocity limit = this x URDF velocity limit")
ap.add_argument("--clip-feedback", type=int, default=1, help="(real_prep3) with --acc-clip 1: write the sent point into the solver state filter")
ap.add_argument("--js-qos", default="reliable", choices=["reliable", "best_effort"], help="QoS of the joint-state subscription (best_effort = rclpy qos_profile_sensor_data)")
ap.add_argument("--link", default="local", choices=["local", "c6"], help="(real_prep4) c6: control-PC gateway topics / QoS / message format")
ap.add_argument("--stale-ms", type=float, default=0.0, help="(real_prep4) stop for good if no joint state for this long; 0 = off")
ap.add_argument("--human", default="off", choices=["off", "shm"], help="(real_prep5 H1) shm: human input from rp5_human_relay.py shared memory; off = as before")
ap.add_argument("--human-shm", default="rp5_human", help="(H1) shared memory name written by rp5_human_relay.py")
ap.add_argument("--human-frame", default="", help="(H1) node world -> robot base frame file (live_frame yml); required with --human shm")
ap.add_argument("--wait-human", type=float, default=120.0, help="(H1) s to wait for the first human record before the warm-up")
ap.add_argument("--skip-late", type=int, default=0, help="(H2) 1: a step that starts after the next slot has passed drops the overdue slots and goes on at the next slot (no backlog); the run still ends at slot N; 0 = off (as before)")
ap.add_argument("--human-near", type=int, default=1, help="(H1) 1: record the minimum gap robot sphere - human shape per step after publishing; (OVERNIGHT_H1) N > 1: every N-th step, 0: off")
ap.add_argument("--rinert-every", default="1", help="(OVERNIGHT_H1) robot inverse inertia update: 1 = every step (as before), N = every N-th step, new = on steps with new human data (at least every 4th step)")
ap.add_argument("--hsel", default="off", choices=["off", "topk", "dist"], help="(OVERNIGHT_H1) human shapes used by the effective-mass cost: off = all (as before); topk = the K segments with the shortest time to collision (gap minus 0.25 m over human closing speed + 0.25 m/s); dist = shapes within R of the robot spheres (at least the nearest segment)")
ap.add_argument("--hsel-k", type=int, default=3, help="(OVERNIGHT_H1) K for --hsel topk")
ap.add_argument("--hsel-r", type=float, default=1.0, help="(OVERNIGHT_H1) R (m) for --hsel dist, from the robot spheres at the measured joints")
ap.add_argument("--hsel-every", default="new", choices=["new", "step"], help="(OVERNIGHT_H1) select on new human data or every step")
ap.add_argument("--reuse-fk", type=int, default=0, help="(OVERNIGHT_H1) 1: the goal / near check after publishing reuses the speed limiter's kinematics of the measured joints (same input) instead of computing them again")
ap.add_argument("--meff-noop", type=int, default=0, help="(OVERNIGHT_H1, profiling only) 1: the effective-mass kernel returns at once (no human cost)")
ap.add_argument("--out-lpf", default="none", help="(filter) smoothing of the published first point: none = off (as before), ema, bw2, cd2, sg, blend, chains a+b")
ap.add_argument("--out-lpf-hz", type=float, default=12.0, help="(filter) cut-off Hz for ema / bw2 / cd2")
ap.add_argument("--out-lpf-n", type=int, default=9, help="(filter) sg: number of targets")
ap.add_argument("--out-lpf-deg", type=int, default=1, help="(filter) sg: polynomial degree")
ap.add_argument("--out-lpf-m", type=int, default=4, help="(filter) blend: number of plans")
ap.add_argument("--out-lpf-at", default="post", choices=["pre", "post"], help="(filter) smoothing before (pre) or after (post) the sphere speed limiter")
ap.add_argument("--out-lpf-feedback", type=int, default=1, help="(filter) 1: write the smoothed point into the solver state filter")
ap.add_argument("--mpc-override", default="", help="(filter) yml merged into the MPC task settings; empty = off")
ap.add_argument("--lim-ref", default="meas", choices=["meas", "cmd"], help="(filter) sphere speed limiter reference: meas = measured joints (as before), cmd = previously published point")
ap.add_argument("--lim-v", type=float, default=0.0, help="(filter) with --lim-ref cmd: sphere speed limit of the commanded path, m/s (0 = profile sphere_speed_limit)")
ap.add_argument("--lim-reuse", type=int, default=1, help="(filter) with --lim-ref cmd: 1 = reuse the previous step's spheres of the published point (no extra kinematics call)")
ap.add_argument("--lim-guard", type=float, default=3.0, help="(filter) with --lim-ref cmd: also keep every sphere within guard x limit x DT of the measured one (0 = off)")
ap.add_argument("--rec-goal-err", type=int, default=0, help="(filter) 1: record goal_pe / goal_qe / raw_q per step and the out_filter summary")
ap.add_argument("--rec-human", type=int, default=0, help="(H2 rec) 1: with --human shm, record the human data the MPC used -> <out>_human.npz")
ap.add_argument("--human-jump-filter", type=int, default=0, help="(H2 jf) 1: reject human records whose torso jumps (rp5_jump_filter.py)")
ap.add_argument("--hjf-jump", type=float, default=None, help="(H2 jf) torso jump per record that starts a hold, m (default 0.2; 0.3 with --hjf-mode anchor)")
ap.add_argument("--hjf-mode", default="hold", choices=["hold", "anchor"], help="(H2 jf) hold = the first rule (default); anchor = rp5_jump_filter.AnchorFilter")
ap.add_argument("--hjf-settle-n", type=int, default=3, help="(H2 jf anchor) records in a row with a small torso step that take a new place")
ap.add_argument("--hjf-settle-step", type=float, default=0.1, help="(H2 jf anchor) small torso step per record, m")
ap.add_argument("--hjf-near", type=float, default=0.25, help="(H2 jf anchor) a new place this close (m) to a robot sphere surface is not taken by settling (0 = no test)")
ap.add_argument("--hjf-hold-max", type=float, default=3.0, help="(H2 jf anchor) longest hold, s (then the record is taken as is)")
ap.add_argument("--hjf-back", type=float, default=0.2, help="(H2 jf) torso back within this of the last accepted one ends the hold, m")
ap.add_argument("--hjf-hold", type=float, default=1.5, help="(H2 jf) longest hold, s (then the new position is taken)")
ap.add_argument("--human-shift", default="", help="(cam20) human record translation dx,dy,dz in the node world frame, m (empty = none)")
ap.add_argument("--arrive-pos-tol", type=float, default=None, help="(arrtol) arrival position tolerance of the goal switch, m (default POS_TOL_M 0.01)")
ap.add_argument("--leg-timeout", type=float, default=None, help="(cam20) leg timeout, s (default: LEG_MAX_STEPS 400 steps = 6.67 s)")
ap.add_argument("--human-robot-exclude", default="", help="(robex) comma list of robot links whose spheres are left out of the human cost (e.g. link0,link1; empty = off)")
ap.add_argument("--human-base-exclude", default="", help="(baseex) yaml of the robot base / cart box: records with the pelvis inside are not a person (empty = off)")
ap.add_argument("--human-vel-clean", type=int, default=0, help="(H2 cost) 1: clip human segment velocities, 0 for jump-filter-rejected records, EMA restart")
ap.add_argument("--hvc-max", type=float, default=2.0, help="(H2 cost) segment speed above this (m/s) is set to 0")
ap.add_argument("--human-d-base", type=float, default=None, help="(H2 cost) human cost base clearance d_base, m (default 0.065)")
ap.add_argument("--human-gap-slow", type=int, default=0, help="(H2 gap) 1: lower the sphere speed cap while human data is rejected or old")
ap.add_argument("--hgs-v", type=float, default=0.14, help="(H2 gap) sphere speed cap while the human data is missing, m/s")
ap.add_argument("--hgs-age", type=float, default=0.1, help="(H2 gap) human data older than this (s) counts as missing")
ap.add_argument("--hgs-rate", type=float, default=2.0, help="(H2 gap) largest change of the cap, m/s per s")
ap.add_argument("--human-field", type=int, default=0, help="(H2 field) 1: effective-mass risk field (outer hill + inner wall)")
ap.add_argument("--hf-wt", type=float, default=0.0, help="(H2 field) field height W (cost units, as the pose cost)")
ap.add_argument("--human-depth-margin", type=float, default=0.0, help="(H2 field) torso depth margin D, m (0 = off)")
ap.add_argument("--hdm-camera", default="", help="(H2 field) camera pose yml for the depth margin (t = camera position, world)")
ap.add_argument("--human-k-gain", type=float, default=None, help="(night E2) mass-dependent part k_gain of the gain_part cost (default: 0.10 as before; 0 = uniform clearance d_base)")
ap.add_argument("--human-eta-max", type=float, default=None, help="(kgain) upper bound eta_max of the clearance thickness, m (default 0.25 as before; raised to d_base if below)")
ap.add_argument("--human-other-d-base", type=float, default=None, help="(kgain) base clearance of the non-arm segments, m (default: not given = d_base, as the arms)")
ap.add_argument("--rec-plan", type=int, default=0, help="(recplan 1124) 1: record the MPC plan up to 0.5 s ahead (steps npz plan_q, plan_t)")
ap.add_argument("--human-mass-mode", default="reduced", choices=["reduced", "human"], help="(massmode 1124) eta reference: reduced (mu, as before) | human (m_h only)")
ap.add_argument("--human-shoulder", type=int, default=0, help="(shoulder 1124) 1: shoulder spheres + shoulders as arm parts + ISO reference 3 kg (see the docstring)")
ap.add_argument("--human-shoulder-json", default="", help="(shoulder 1124) shoulder spheres (default goal1/data/shapes_v2_shoulder.json)")
ap.add_argument("--human-iso-override-json", default="", help="(shoulder 1124) ISO reference override (default goal1/data/iso_ref_override.json)")
ap.add_argument("--human-other-k-gain", type=float, default=None, help="(kgain) mass-dependent part of the non-arm segments (default: not given = k_gain, as the arms)")
ap.add_argument("--human-method", default="", choices=["", "effmass", "uniform"], help="(exp1121) avoidance method of the human cost, same risk field and other options for both: effmass = effective-mass clearance (k_gain 0.10, as the default); uniform = the effmass-mpc comparison baseline, k_gain 0 (clearance d_base in every direction, field d_base .. 2 d_base); written to the summary (human.method). Not given = as before (k_gain from --human-k-gain or 0.10, no method key)")
ap.add_argument("--infeas-escape", type=int, default=0, help="(night task 1) 1: on an infeasible step with the MEASURED state inside the self-collision buffer (self term > 0), move the joints one small step down the self-collision gradient (central differences, --esc-step rad, published only if the world term stays 0 and the joint limits keep 0.05 rad) instead of holding; the MPC takes over when feasible again. 0 = hold as before")
ap.add_argument("--esc-step", type=float, default=0.004, help="(night task 1) joint step norm per 1/60 s step of the escape, rad (0.004 = 0.24 rad/s)")
ap.add_argument("--goal-cycle", type=int, default=0, help="(demo) 1: visit all goals of the scene in order (0 = A <-> B toggle, as before)")
ap.add_argument("--via-tol", type=float, default=0.03, help="(demo) switch tolerance, m, of the goals listed in the scene key 'via' (with --goal-cycle 1)")
ap.add_argument("--via-rot-tol", type=float, default=0.10, help="(demo) switch orientation tolerance, rad, of the 'via' goals (with --goal-cycle 1)")
ap.add_argument("--speed-cap-dist", type=float, default=None, help="(H2 cost) override the profile's speed_cap_dist, m (default: profile)")
ap.add_argument("--rec-camera-pose", default="", help="(H2 rec) camera pose yml kept (path + text) in <out>_human.npz")
args = ap.parse_args()
if args.human != "off" and not args.human_frame:
    ap.error("--human shm needs --human-frame")
if args.js_topic is None:
    args.js_topic = "/c6/joint_states" if args.link == "c6" else "/NS_1/franka/joint_states"
C6 = args.link == "c6"; STALE = args.stale_ms / 1000.0; STOP = {}
os.environ["STEP_ROBOT"] = "fr3v2_1"
import step3_ros_common as RC
RC.add_ros_paths()
import numpy as np, torch, yaml
import step2_scene as S
log = lambda *a: print("[rp2]", *a, flush=True)
PF = yaml.safe_load(open(os.path.expanduser(args.profile))); SET = PF.get("settings", {})
if SET.get("robot_cfg"):
    S.ROBOT_CFG_OVERRIDE = os.path.expanduser(SET["robot_cfg"])
S.POSE_ROT_W = SET.get("pose_rot_w"); S.MPPI_SEED = args.mppi_seed
if int(SET.get("mppi_particles", 0) or 0) > 0 or int(SET.get("mppi_iters", 0) or 0) > 0:   # (H2 / OVERNIGHT_H1) profile keys: MPPI num_particles, n_iters (particle_mpc.yml 400, 5 otherwise)
    S.MPPI_SIZE = {**({"num_particles": int(SET["mppi_particles"])} if int(SET.get("mppi_particles", 0) or 0) > 0 else {}),
                   **({"n_iters": int(SET["mppi_iters"])} if int(SET.get("mppi_iters", 0) or 0) > 0 else {})}
SPEED_LIM = float(SET.get("sphere_speed_limit", 0.0))
NOT_APPLIED = sorted(k for k in SET if k.startswith(("speed_cap", "meff", "human")))
HUMAN_ON = args.human != "off"
if args.human_shoulder and not HUMAN_ON:   # (shoulder 1124)
    raise SystemExit("--human-shoulder 1 needs the human channel (--human shm)")
if HUMAN_ON:   # (H1) these profile keys are used by the human channel below
    NOT_APPLIED = sorted(k for k in NOT_APPLIED if k not in ("speed_cap", "speed_cap_v", "speed_cap_dist", "speed_cap_w", "meff_motion", "human_vel_alpha",
                                                             "human_loss_hold_s", "human_loss_detect_s"))
S.G2_SCENE_JSON = os.path.expanduser(args.scene_json); SCENE = S.set_scene("g2")
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
setup_curobo_logger("warn")
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, qos_profile_sensor_data, ReliabilityPolicy
BE1 = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)   # (real_prep4) as traj_sender.py and the gateway
from sensor_msgs.msg import JointState as RosJS
from trajectory_msgs.msg import JointTrajectory as RosJT, JointTrajectoryPoint as RosJTP
from builtin_interfaces.msg import Duration as RosDuration
ta = TensorDeviceType(); t0 = time.perf_counter()
MPC_OVR = None
if args.mpc_override:   # (filter) merge the given yml into the override file build_mpc passes to the MPC config loader
    def _merge(a, b):
        for k_, v_ in a.items():
            if isinstance(v_, dict) and isinstance(b.get(k_), dict):
                _merge(v_, b[k_])
            else:
                b[k_] = v_
        return b
    _orig_ovr = S.mppi_override_file

    def _ovr_file():
        _b = _orig_ovr(); _d = yaml.safe_load(open(_b)) if _b else {}
        _d = _merge(yaml.safe_load(open(os.path.expanduser(args.mpc_override))) or {}, _d or {})
        _f = args.out + "_mpc_override.yml"; yaml.safe_dump(_d, open(_f, "w")); MPC_OVR_D.update(file=_f, merged=_d); return _f
    MPC_OVR_D = {}; S.mppi_override_file = _ovr_file
mpc = S.build_mpc(S.build_robot_cfg(), S.build_world_config()); torch.cuda.synchronize()
if args.mpc_override:
    MPC_OVR = dict(MPC_OVR_D, src=args.mpc_override, sample_mode=str(mpc.solver.optimizers[0].sample_mode), init_cov=float(getattr(mpc.solver.optimizers[0], "init_cov", float("nan"))))
    log(f"(filter) MPC settings override: {MPC_OVR}")
_sb = mpc.rollout_fn.dynamics_model.get_state_bounds()
JLIM = dict(velocity=[round(float(x), 3) for x in _sb.velocity[1].reshape(-1)], acceleration=[round(float(x), 3) for x in _sb.acceleration[1].reshape(-1)])
log(f"MPC built {time.perf_counter() - t0:.1f}s, profile {args.profile}, robot cfg {S.ROBOT_CFG_OVERRIDE or 'default'}, limits {JLIM}, sphere speed limit {SPEED_LIM}, "
    f"profile keys not applied (no human channel): {NOT_APPLIED}")
NJ = len(mpc.joint_names); NAMES = [f"fr3_joint{k + 1}" for k in range(NJ)]; DTF = float(mpc.solver.safety_rollout.dynamics_model.state_filter.dt)
TRAJ_DT = mpc.rollout_fn.dynamics_model.traj_dt.reshape(-1); HT = torch.cumsum(TRAJ_DT, 0); HT_NP = HT.cpu().numpy()
HUMAN = MEFF = MEFF_INP = RI = None; HINFO = {}; HSH = None   # (shoulder 1124) HSH = human_shoulder.ShoulderSpheres with --human-shoulder 1
if HUMAN_ON:   # (H1) effective-mass / speed-cap channel as in step2b --meff on (live runs); hooked in before the goal setup and the first solve, as step2b
    sys.path.insert(0, os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/goal1/scripts"))
    if args.meff_noop:   # (OVERNIGHT_H1) profiling only: g1_meff_gpu reads G1_MEFF_NOOP when the channel is built
        os.environ["G1_MEFF_NOOP"] = "1"
    import g1_meff_gpu as G1M, g1_human_link as G1L
    from effmass_robot.robot_inertia_np import RobotInertiaNP
    _ls = mpc.kinematics.kinematics_config.link_spheres.cpu().numpy(); MEFF_SV = int((_ls[:, 3] > 0).sum())
    assert (_ls[:MEFF_SV, 3] > 0).all() and not (_ls[MEFF_SV:, 3] > 0).any()
    MEFF_INP = G1M.MeffInputs(MEFF_SV, S.describe_mpc(mpc)["traj_dt"], device=str(ta.device))
    RI = RobotInertiaNP(rotor=True); assert len(RI.sphere_link) == MEFF_SV
    _rng = np.random.default_rng(5); _lo = np.array([-2.7, -1.7, -2.8, -3.0, -2.7, 0.6, -3.0]); _hi = -_lo; _hi[3] = -0.2; _hi[5] = 4.4; _nz = np.zeros(MEFF_SV)
    for _ in range(20):   # spheres whose robot inverse inertia is always 0 (as step2b)
        _nz = np.maximum(_nz, np.abs(RI.compute(_lo + (_hi - _lo) * _rng.uniform(size=7))[1]).reshape(MEFF_SV, -1).max(1))
    MEFF_MASS_ON = (_nz > 0).astype(np.int32)
    HX_LINKS = [x.strip() for x in args.human_robot_exclude.split(",") if x.strip()]   # (robex 1124) spheres left out of the human cost
    HX = [i for i, n_ in enumerate(RI.sphere_link) if any(str(n_) == l_ or str(n_).endswith("_" + l_) for l_ in HX_LINKS)]
    if HX_LINKS and not HX:
        raise SystemExit(f"--human-robot-exclude {args.human_robot_exclude}: no robot sphere of these links ({sorted(set(map(str, RI.sphere_link)))})")
    HX_KEEP = np.array([i for i in range(MEFF_SV) if i not in HX], int)
    if HX:
        _hxc = {}
        for i in HX:
            _hxc[str(RI.sphere_link[i])] = _hxc.get(str(RI.sphere_link[i]), 0) + 1
        HXD = dict(links=HX_LINKS, spheres=HX, n=len(HX), of=MEFF_SV, per_link=_hxc)
        log(f"(robex) human cost robot spheres left out: {len(HX)} of {MEFF_SV} ({', '.join('%s x%d' % kv for kv in _hxc.items())}; indices {HX})")
    HP = dict(speed_cap=str(SET.get("speed_cap", "off")), speed_cap_v=float(SET.get("speed_cap_v", 0.5)), speed_cap_dist=float(SET.get("speed_cap_dist", 0.4)),
              speed_cap_w=float(SET.get("speed_cap_w", 0.0)), meff_motion=str(SET.get("meff_motion", "static")), human_vel_alpha=float(SET.get("human_vel_alpha", 0.0)),
              human_loss_hold_s=float(SET.get("human_loss_hold_s", 0.0)), human_loss_detect_s=float(SET.get("human_loss_detect_s", 0.1)))
    if args.speed_cap_dist is not None:   # (H2 cost) range of the speed cost from the command line
        log(f"(H2 cost) speed_cap_dist {HP['speed_cap_dist']} -> {args.speed_cap_dist} m (--speed-cap-dist)"); HP["speed_cap_dist"] = float(args.speed_cap_dist)
    LIVE_COST = dict(mode="gain_part", agg="max_all", w=100000.0, eta0=0.065, muref=0.53, p=0.10, eta_min=0.065, eta_max=0.25)   # lv_live_kinect.sh values
    if args.human_d_base is not None:   # (H2 cost) base clearance from the command line
        LIVE_COST.update(eta0=float(args.human_d_base), eta_min=float(args.human_d_base), eta_max=max(0.25, float(args.human_d_base)))
        log(f"(H2 cost) human cost d_base 0.065 -> {args.human_d_base} m (--human-d-base)")
    if args.human_eta_max is not None:   # (kgain) thickness upper bound
        LIVE_COST.update(eta_max=max(float(args.human_eta_max), LIVE_COST["eta0"])); log(f"(kgain) human cost eta_max -> {LIVE_COST['eta_max']} m (--human-eta-max {args.human_eta_max})")
    if args.human_k_gain is not None:   # (night E2) 0 = uniform clearance
        LIVE_COST.update(p=float(args.human_k_gain)); log(f"(night) human cost k_gain 0.10 -> {args.human_k_gain} (--human-k-gain)")
    if args.human_method:   # (exp1121) uniform = k_gain 0 (effmass-mpc baseline), effmass = k_gain 0.10
        _kg = 0.0 if args.human_method == "uniform" else (0.10 if args.human_k_gain is None else float(args.human_k_gain))   # (kgain) effmass: K of --human-k-gain
        if args.human_k_gain is not None and abs(float(args.human_k_gain) - _kg) > 1e-12:
            raise SystemExit(f"--human-method {args.human_method} (k_gain {_kg}) and --human-k-gain {args.human_k_gain} differ: give one")
        LIVE_COST.update(p=_kg); log(f"(exp1121) human method {args.human_method}: k_gain {_kg}, d_base {LIVE_COST['eta0']}")
    if args.human_mass_mode == "human":   # (massmode 1124) kernel mode gain_part_h: same parameters, reference = the human effective mass
        LIVE_COST.update(mode="gain_part_h"); log("(massmode) human cost reference: human effective mass m_h / m_iso (gain_part_h)")
    MEFF = G1M.MeffChannel(mpc, MEFF_INP, impl="warp", direction="closest", motion=HP["meff_motion"], hfront=0, mass_on=MEFF_MASS_ON, record_q=False,
                           speed_cap=HP["speed_cap"] == "on", shape_select=args.hsel != "off", human_exclude=HX or None)   # (robex 1124)
    MEFF.set_cost(**LIVE_COST); MEFF.set_speed_cap(v=HP["speed_cap_v"], dist=HP["speed_cap_dist"], w=HP["speed_cap_w"])
    if args.human_shoulder:   # (shoulder 1124) spheres on the records, shoulders = arm parts, ISO reference override
        from human_shoulder import ShoulderSpheres
        HSH = ShoulderSpheres(args.human_shoulder_json, args.human_iso_override_json)
        MEFF.set_parts(tuple(G1M.ARM_PARTS) + tuple(G1M.SHOULDER_PARTS), HSH.iso); log(f"(shoulder) human shoulder on: {HSH.describe()}")
    if args.human_other_d_base is not None or args.human_other_k_gain is not None:   # (kgain) non-arm segments: own d0 / k_gain
        if args.human_other_d_base is not None and args.human_other_d_base > LIVE_COST["eta_max"]:
            raise SystemExit(f"--human-other-d-base {args.human_other_d_base} > eta_max {LIVE_COST['eta_max']}: raise --human-eta-max")
        MEFF.set_other(eta0=args.human_other_d_base, p=args.human_other_k_gain); log(f"(kgain) non-arm segments: d0 {args.human_other_d_base}, k_gain {args.human_other_k_gain} (None = as the arms)")
    HMETA = dict(human_k_gain=float(LIVE_COST["p"]), human_eta_max=float(LIVE_COST["eta_max"]),   # (kgain) values used (also without the options)
                 human_other_d_base=float(LIVE_COST["eta0"] if args.human_other_d_base is None else args.human_other_d_base),
                 human_other_k_gain=float(LIVE_COST["p"] if args.human_other_k_gain is None else args.human_other_k_gain), human_arm_parts=[G1M.ISO_SEGMENTS[g] for g in MEFF.arm_segs], human_mass_mode=args.human_mass_mode)   # (massmode 1124)
    if HSH is not None:   # (shoulder 1124) only with the option (the keys are absent otherwise, as before)
        HMETA.update(human_shoulder=1, human_iso_override=json.dumps(HSH.iso, sort_keys=True), human_shoulder_shapes=HSH.shapes_json)
    log(f"(kgain) human cost parts: {HMETA}")
    if args.human_field:   # (H2 field)
        MEFF.set_field(1, args.hf_wt); log(f"(H2 field) risk field on: W {args.hf_wt}, L = d_req (outer hill d_req .. 2 d_req)")
    if args.hsel != "off":   # (OVERNIGHT_H1) shape selection: wrap MeffInputs.set_human (reorder: selected shapes first, then set the count)
        HSEL_SET = MEFF_INP.set_human; HSEL = dict(h=None, mask=np.ones(G1M.N_SHAPES, bool), n=G1M.N_SHAPES, sph=None)

        def hsel_apply(h):
            sph = HSEL["sph"] if (HSEL["sph"] is None or not HX) else HSEL["sph"][HX_KEEP]   # (robex 1124) left-out spheres do not pick shapes
            rad = np.asarray(h["rad"], np.float64); seg = np.asarray(h["seg"]).astype(int); act = rad > 0
            if sph is None:
                sel = act.copy()
            else:
                p0 = np.asarray(h["p0"], np.float64); d = np.asarray(h["p1"], np.float64) - p0; c = sph[:, None, :3]
                t = np.clip(((c - p0[None]) * d[None]).sum(-1) / np.maximum((d * d).sum(-1), 1e-12)[None], 0, 1)
                q = p0[None] + t[..., None] * d[None]
                if MEFF_INP.hdm is not None:   # (H2 field) depth margin: swept torso shapes, as the kernel
                    q = G1M.swept_closest(c, p0[None], d[None], MEFF_INP.margin_for(h)[None])
                g = np.linalg.norm(q - c, axis=-1) - rad[None] - sph[:, None, 3]   # (S,24)
                j = g.argmin(0); gs = np.where(act, g[j, np.arange(len(rad))], np.inf)
                segs = np.unique(seg[act]); gseg = np.array([gs[seg == sg].min() for sg in segs])
                if args.hsel == "topk":
                    u = c[j, 0] - q[j, np.arange(len(rad))]; u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)   # human -> robot, per shape
                    vh = np.maximum((np.asarray(h["vel"], np.float64)[seg] * u).sum(-1), 0.0)
                    vseg = np.array([vh[seg == sg].max() for sg in segs])
                    ttc = np.maximum(gseg - 0.25, 0.0) / (vseg + 0.25)
                    keep = segs[np.lexsort((gseg, ttc))[:max(1, args.hsel_k)]]
                    sel = act & np.isin(seg, keep)
                else:
                    sel = act & (gs < args.hsel_r)
                    if not sel.any():
                        sel = act & (seg == segs[int(np.argmin(gseg))])
            order = np.concatenate([np.nonzero(sel)[0], np.nonzero(~sel)[0]])
            h2 = dict(h); h2.update({k: np.asarray(h[k])[order] for k in ("p0", "p1", "rad", "seg")})
            HSEL_SET(h2); MEFF_INP.set_nact(int(sel.sum())); HSEL["mask"] = sel; HSEL["n"] = int(sel.sum())

        def hsel_set_human(h):
            HSEL["h"] = h; hsel_apply(h)
        MEFF_INP.set_human = hsel_set_human
    HFRAME = os.path.expanduser(args.human_frame)
    HUMAN = G1L.HumanInertiaLink(None, HFRAME, MEFF_INP, stage=None, display=(), shapes_viz=False, vel_alpha=HP["human_vel_alpha"],
                                 loss_hold_s=HP["human_loss_hold_s"], loss_detect_s=HP["human_loss_detect_s"], gui=False)
    HDM_INFO = None
    if args.human_depth_margin > 0:   # (H2 field) camera position in the robot base frame
        import yaml as _yaml
        _cf = os.path.expanduser(args.hdm_camera or args.rec_camera_pose or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "camera_pose_rp5.yaml"))
        _cw = np.asarray(_yaml.safe_load(open(_cf))["t"], np.float64); _cb = np.asarray(HUMAN.R, np.float64) @ _cw + np.asarray(HUMAN.t, np.float64)
        MEFF_INP.set_depth_margin(args.human_depth_margin, _cb)
        HDM_INFO = dict(D_m=args.human_depth_margin, segments=["pelvis", "abdomen", "chest"], camera_file=os.path.abspath(_cf), camera_world=_cw.tolist(), camera_base=_cb.tolist(),
                        method="swept along the camera ray (exact closest point)")
        log(f"(H2 field) torso depth margin on: {HDM_INFO}")
    HINFO = dict(source="shm", shm=args.human_shm, frame=HFRAME, R=HUMAN.R.tolist(), t=HUMAN.t.tolist(), spheres=MEFF_SV,
                 zero_inertia_spheres=np.nonzero(MEFF_MASS_ON == 0)[0].tolist(), profile_human=HP, cost=MEFF.cost_state(), optimizers=MEFF.info)
    HINFO["part_cost"] = dict(HMETA, per_segment=MEFF.part_state())   # (kgain)
    if args.human_method:   # (exp1121)
        HINFO["method"] = args.human_method
    log(f"(H1) human channel: {HINFO}")
q_start = torch.tensor(SCENE["q_start"], **ta.as_torch_dict()).reshape(1, -1)
js0 = JointState.from_position(q_start.clone(), joint_names=mpc.joint_names)
goal_idx = 1
APT = S.POS_TOL_M if args.arrive_pos_tol is None else float(args.arrive_pos_tol)   # (arrtol) arrival position tolerance, m
GC = None
if args.goal_cycle:   # (demo) waypoint cycle: tolerances per goal, arrival log
    _via = set(int(v) for v in json.load(open(os.path.expanduser(args.scene_json))).get("via", []))
    _ng = len(S.GOALS_WXYZ); assert _ng >= 2 and all(0 <= v < _ng for v in _via), f"--goal-cycle: {_ng} goals, via {sorted(_via)}"
    GC = dict(n=_ng, via=sorted(_via), tol=[(args.via_tol, args.via_rot_tol) if g in _via else (APT, S.ROT_TOL_RAD) for g in range(_ng)],
              arr=[0] * _ng, timeouts=[0] * _ng, lap_steps=[])
    log(f"(demo) goal cycle: {_ng} goals, via {GC['via']} at {args.via_tol} m / {args.via_rot_tol} rad, others {APT} m / {S.ROT_TOL_RAD} rad")
gb = mpc.setup_solve_single(Goal(current_state=js0, goal_state=js0, goal_pose=Pose.from_list(S.GOALS_WXYZ[goal_idx], ta)), 1)
mpc.update_goal(gb); torch.cuda.synchronize()
# ---- ROS ----
rclpy.init(); node = rclpy.create_node("rp2_mpc_rt"); LAST = [None, 0, 0.0, 0.0]   # (msg arrays, receive perf time, header stamp s, receive wall s)
NRECV = [0]


def on_js(m):
    idx = {n: i for i, n in enumerate(m.name)}
    try:
        q = np.array([m.position[idx[n]] for n in NAMES]); v = np.array([m.velocity[idx[n]] for n in NAMES]) if len(m.velocity) == len(m.name) else np.zeros(NJ)
    except KeyError:
        return
    LAST[0] = (q, v); LAST[1] = time.perf_counter(); LAST[2] = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec; LAST[3] = time.time(); NRECV[0] += 1


SHM = None; SEQ = [-1.0]
if args.js_source == "shm":   # (test a) the joint state comes from rp2_js_relay.py through shared memory
    from multiprocessing import shared_memory, resource_tracker
    w0 = time.time()
    while SHM is None and time.time() - w0 < args.wait_js:
        try:
            SHM = shared_memory.SharedMemory(name=args.shm_name)
        except FileNotFoundError:
            time.sleep(0.05)
    if SHM is None:
        log(f"no shared memory {args.shm_name}"); sys.exit(3)
    resource_tracker.unregister(SHM._name, "shared_memory")   # the relay owns and removes the block
    SB = np.ndarray((18 if SHM.size >= 18 * 8 else 17,), np.float64, buffer=SHM.buf)   # index 17: receive time, monotonic (real_prep4 relay)


def read_shm():
    for _ in range(100):
        r = SB.copy()
        if r[0] % 2 == 0 and r[0] == SB[0]:
            break
    if r[0] <= 0:
        return
    if r[0] != SEQ[0]:
        NRECV[0] += int((r[0] - max(SEQ[0], 0)) / 2); SEQ[0] = r[0]
    LAST[0] = (r[3:10].copy(), r[10:17].copy()); LAST[1] = r[17] if len(r) > 17 and r[17] > 0 else time.perf_counter() - (time.time() - r[2]); LAST[2] = r[1]; LAST[3] = r[2]


if args.js_source == "topic":
    node.create_subscription(RosJS, args.js_topic, on_js, BE1 if C6 else qos_profile_sensor_data if args.js_qos == "best_effort" else QoSProfile(depth=1))
pub = node.create_publisher(RosJT, args.jt_topic, BE1 if C6 else QoSProfile(depth=10))
ex = SingleThreadedExecutor(); ex.add_node(node); threading.Thread(target=ex.spin, daemon=True).start()
w0 = time.time()
while LAST[0] is None and time.time() - w0 < args.wait_js:
    time.sleep(0.01)
    if args.js_source == "shm":
        read_shm()
if LAST[0] is None:
    log(f"no joint state on {args.js_topic} within {args.wait_js} s"); sys.exit(3)
log(f"first joint state after {time.time() - w0:.2f}s on {args.js_topic}")


HSHM = None; HSEQ = [-1.0]; HLEN = 509   # (H1) rp5_human_relay.py record: 8 header values + com 57, vel 57, inv 171, p0 72, p1 72, rad 24, seg 24, type 24
HSHIFT = None   # (cam20) --human-shift (node world frame)
if args.human_shift:
    HSHIFT = np.array([float(x) for x in args.human_shift.split(",")], np.float64); assert HSHIFT.shape == (3,), "--human-shift dx,dy,dz"
    if not np.any(HSHIFT):
        HSHIFT = None
LEG_STEPS = S.LEG_MAX_STEPS if args.leg_timeout is None else max(1, int(round(args.leg_timeout * 60)))   # (cam20) leg timeout, steps
HOFF = dict(com=(8, 57), vel=(65, 57), inv=(122, 171), p0=(293, 72), p1=(365, 72), rad=(437, 24), seg=(461, 24), type=(485, 24))
HREC = dict(relay_msgs=0.0, pickups=0)
HRECL = [] if (HUMAN_ON and args.rec_human) else None; HSTEP = [-1]   # (H2 rec) picked-up records (step, pickup wall time, raw record)
if args.rec_human and not HUMAN_ON:
    log("(H2 rec) --rec-human ignored: no --human shm")
JF = None; JF_ACC = [None]   # (H2 jf) filter, last accepted raw record
HGS_REJ = [0]   # (H2 gap) 1 = the last human record read was rejected by the jump filter
HVC = dict(records=0, clipped_segments=0, clipped_records=0, zeroed_records=0, ema_restarts=0, prev_rej=0, vmax_seen=0.0) if args.human_vel_clean else None   # (H2 cost)
if HUMAN_ON and args.human_jump_filter:
    from rp5_jump_filter import JumpFilter, AnchorFilter, robot_near_fn
    if args.hjf_mode == "anchor":   # robot spheres at the latest measured joints; RI / HUMAN.R, t are set up below, read at call time
        _hjf_near = lambda tor: robot_near_fn(RI, lambda: LAST[0][0] if LAST[0] is not None else None, HUMAN.R, HUMAN.t)(tor)
        JF = AnchorFilter(0.3 if args.hjf_jump is None else args.hjf_jump, args.hjf_back, args.hjf_settle_n, args.hjf_settle_step, args.hjf_near,
                          args.hjf_hold_max, near_fn=_hjf_near if args.hjf_near > 0 else None)
    else:
        JF = JumpFilter(0.2 if args.hjf_jump is None else args.hjf_jump, args.hjf_back, args.hjf_hold)
    log(f"(H2 jf) human jump filter on: {JF.describe()}")
BEX = None; BEX_ACC = [None]; BEX_N = dict(seen=0, excluded=0, no_previous=0); BEX_T = []; BEX_FL = []   # (baseex) rule, last used record, counters, stamps, flags
if args.human_base_exclude and not HUMAN_ON:
    log("(baseex) --human-base-exclude ignored: no --human shm")
elif args.human_base_exclude:
    from rp5_base_exclude import BaseExclude
    BEX = BaseExclude(args.human_base_exclude); log(f"(baseex) human base exclude on: {BEX.describe()}")
if HUMAN_ON:
    from multiprocessing import shared_memory as _hsm, resource_tracker as _hrt
    w0 = time.time()
    while HSHM is None and time.time() - w0 < args.wait_human:
        try:
            HSHM = _hsm.SharedMemory(name=args.human_shm)
        except FileNotFoundError:
            time.sleep(0.05)
    if HSHM is None:
        log(f"(H1) no shared memory {args.human_shm} within {args.wait_human} s (start rp5_human_relay.py first)"); sys.exit(3)
    _hrt.unregister(HSHM._name, "shared_memory")   # the relay owns and removes the block
    HB = np.ndarray((HLEN,), np.float64, buffer=HSHM.buf)


def read_human():
    """(H1) copy the latest relay record (seqlock) into HumanInertiaLink's latest slot, with the same fields as its ROS callback _on_msg."""
    for _ in range(100):
        r = HB.copy()
        if r[0] % 2 == 0 and r[0] == HB[0]:
            break
    else:
        return
    if r[0] <= 0 or r[0] == HSEQ[0]:
        return
    HSEQ[0] = r[0]; HREC["relay_msgs"] = float(r[7]); HREC["pickups"] += 1
    if HSHIFT is not None:   # (cam20) camera translation correction: COM and shape end points (node world frame)
        for _k, _n in (("com", 19), ("p0", 24), ("p1", 24)):
            _o = HOFF[_k][0]; r[_o:_o + 3 * _n] = (r[_o:_o + 3 * _n].reshape(_n, 3) + HSHIFT).reshape(-1)
    if HSH is not None:   # (shoulder 1124) shoulder spheres into the empty slots of this record copy (used, filtered and recorded as received)
        HSH.apply(r, HOFF)
    _rej = 0   # (H2 jf) a rejected record: data of the last accepted one, stamp / receive time of this one
    _bex = 0   # (baseex) 1 = pelvis inside the base box: not given to the jump filter, handled as rejected
    if BEX is not None:
        BEX_N["seen"] += 1
        if BEX.inside(r[8:11]):
            _bex = 1; BEX_N["excluded"] += 1; BEX_T.append(float(int(r[1])) + int(r[2]) * 1e-9)
            if (JF_ACC[0] if JF is not None else BEX_ACC[0]) is None:   # no earlier person: as no message
                BEX_N["no_previous"] += 1; return
    if JF is not None and not _bex:
        if JF.step(float(int(r[1])) + int(r[2]) * 1e-9, r[8:65].reshape(19, 3)):
            JF_ACC[0] = r
        else:
            _rej = 1
    rd = r if not _rej else JF_ACC[0]
    if BEX is not None:
        if _bex:
            _rej = 1; rd = JF_ACC[0] if JF is not None else BEX_ACC[0]
        elif not _rej:
            BEX_ACC[0] = r
    HGS_REJ[0] = _rej
    g = lambda k: rd[HOFF[k][0]:HOFF[k][0] + HOFF[k][1]]
    st = int(r[1]) * 1_000_000_000 + int(r[2]); t = int(round(r[3] * 1e9))
    _vel = g("vel").astype(np.float32)
    if HVC is not None:   # (H2 cost) velocity clean-up
        HVC["records"] += 1; _v = _vel.reshape(19, 3).copy(); _sp = np.linalg.norm(_v, axis=1); HVC["vmax_seen"] = max(HVC["vmax_seen"], float(_sp.max()))
        if _rej:
            _v[:] = 0.0; HVC["zeroed_records"] += 1
        else:
            _m = _sp > args.hvc_max; _v[_m] = 0.0; HVC["clipped_segments"] += int(_m.sum()); HVC["clipped_records"] += int(_m.any())
            if HVC["prev_rej"]:   # hold ended: restart the velocity EMA (HumanInertiaLink.pump uses vf)
                HUMAN.vf = None; HVC["ema_restarts"] += 1
        HVC["prev_rej"] = _rej; _vel = _v.reshape(-1)
    d = dict(com=g("com").astype(np.float32), vel=_vel, inv=g("inv").astype(np.float32), p0=g("p0").astype(np.float32),
             p1=g("p1").astype(np.float32), rad=g("rad").astype(np.float32), seg=np.rint(g("seg")).astype(np.int64), type=np.rint(g("type")).astype(np.uint8))
    with HUMAN.lock:
        HUMAN.n_dyn += 1
        HUMAN.latest = (HUMAN.n_dyn, st, t, int(r[4]), d)
    HUMAN.recv.append((HUMAN.n_dyn, st, t, int(r[4]), float(r[5]), HOFF["rad"][1]))
    if HRECL is not None:   # (H2 rec) r is already a copy
        HRECL.append((HSTEP[0], time.time(), r, _rej))
        if BEX is not None:   # (baseex) flag per recorded record
            BEX_FL.append(_bex)


def HUMAN_HOOK(step):   # slot for the human input (sensing -> human dynamics -> MPC cost); (H1) used with --human shm, else nothing
    if not HUMAN_ON:
        return None
    HSTEP[0] = step; read_human()
    return HUMAN.pump()


def human_robot_inertia(q):   # (H1) robot inverse inertia at the measured joints -> MeffInputs (as step2b, once per step)
    MEFF_INP.set_robot(RI.compute(np.asarray(q, np.float64).reshape(-1))[1])


def state_from_last(comp=False):
    q, v = LAST[0]
    if comp:   # age compensation: move the measured angles forward by velocity x age (header stamp)
        q = q + v * max(time.time() - LAST[2], 0.0)
    return JointState(position=torch.tensor(q, **ta.as_torch_dict()).reshape(1, -1), velocity=torch.tensor(v, **ta.as_torch_dict()).reshape(1, -1),
                      acceleration=torch.zeros(1, NJ, **ta.as_torch_dict()), jerk=torch.zeros(1, NJ, **ta.as_torch_dict()), joint_names=list(mpc.joint_names))


if HUMAN_ON:   # (H1) first human before the warm-up (as step2b waits for the first message), robot inertia at the measured joints
    w0 = time.time()
    while HUMAN.n_dyn == 0 and time.time() - w0 < args.wait_human:
        read_human(); time.sleep(0.02)
    if HUMAN.n_dyn == 0:
        log(f"(H1) no human record in shared memory {args.human_shm} within {args.wait_human} s (sensing / human node / relay running?)"); sys.exit(3)
    HUMAN.pump()
    if args.js_source == "shm":
        read_shm()
    human_robot_inertia(LAST[0][0])
    _h = HUMAN.last_h; _pel = _h["com"][0]
    log(f"(H1) first human after {time.time() - w0:.2f}s: age {MEFF_INP.age_val * 1000:.0f} ms, pelvis COM (base) {np.round(_pel, 3).tolist()}, "
        f"relay messages so far {int(HREC['relay_msgs'])}")
# ---- warm-up (graph capture) ----
WARM = []
for _ in range(args.warmup):
    torch.cuda.synchronize(); _t = time.perf_counter(); mpc.step(state_from_last(), max_attempts=S.MAX_ATTEMPTS); torch.cuda.synchronize(); WARM.append(round((time.perf_counter() - _t) * 1000, 2))
log(f"warm-up ms {WARM}")
if HUMAN_ON and args.human_near:   # (H1) first query_all call outside the timed loop (its first call took ~90 ms in the smoke test)
    for _ in range(3):
        MEFF.query_all(mpc.rollout_fn.compute_kinematics(state_from_last()).robot_spheres.reshape(-1, 4)[:MEFF_SV])
    torch.cuda.synchronize()
DT = 1.0 / 60.0; N = int(round(args.duration * 60)); pc = time.perf_counter
VLIM = np.array(JLIM["velocity"][:NJ]) * args.vel_frac; SFILT = mpc.solver.safety_rollout.dynamics_model.state_filter
CLIP = dict(p1=None, p2=None)   # (real_prep3) the two targets sent last
if args.acc_clip:
    CLIP["p1"] = LAST[0][0].copy(); CLIP["p2"] = LAST[0][0].copy()
    log(f"acceleration clip on: acc_max {args.acc_max} rad/s^2, velocity limits {np.round(VLIM, 3).tolist()} rad/s, feedback to the state filter {args.clip_feedback}")


def acc_clip(q, qd, qdd):
    """returns the clipped first point (q, qd, qdd), the per-joint change of q and whether anything was changed"""
    p1, p2 = CLIP["p1"], CLIP["p2"]; vp = (p1 - p2) / DT; v = (q - p1) / DT
    lo = np.maximum(vp - args.acc_max * DT, -VLIM); hi = np.minimum(vp + args.acc_max * DT, VLIM)
    lo = np.minimum(lo, hi)   # if the previous velocity was above the velocity limit, decelerate as fast as allowed
    v2 = np.clip(v, lo, hi); q2 = p1 + v2 * DT; ch = bool(np.any(np.abs(v2 - v) > 1e-12))
    return (q2, v2, (v2 - vp) / DT, q2 - q, ch) if ch else (q, qd, qdd, np.zeros_like(q), False)
R = {k: np.zeros(N) for k in ("t_sched", "t_start", "t_state", "t_solve0", "t_solve1", "t_pub", "t_end", "state_age", "stamp_age", "lim_scale", "sphere_v_before")}
R["clip_dq"] = np.zeros((N, NJ), np.float32); R["clipped"] = np.zeros(N, np.int8); R["acc_pre"] = np.zeros((N, NJ), np.float32)
JLO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]); JHI = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])   # (night) fr3v2_1 urdf
if args.infeas_escape:   # (night) records only with the option
    R["inf_term"] = np.zeros(N, np.int8)   # infeasible steps: bit 1 bound, 2 world, 4 self; 8 = escaped (published)
    R["inf_self"] = np.full((N, 2), np.nan, np.float32)   # self-collision value of the command / of the measured state
R["feasible"] = np.zeros(N, np.int8); R["sent_q"] = np.zeros((N, NJ), np.float32); R["sent_qd"] = np.zeros((N, NJ), np.float32); R["sent_qdd"] = np.zeros((N, NJ), np.float32)
NPLAN = int((HT_NP <= 0.5 + 1e-6).sum()) or len(HT_NP)   # (recplan 1124) plan points within 0.5 s (all if the horizon is shorter)
if args.rec_plan:
    R["plan_q"] = np.full((N, NPLAN, NJ), np.nan, np.float32); log(f"(recplan) MPC plan recorded: {NPLAN} points, {HT_NP[NPLAN - 1]:.3f} s ahead")
R["pub_age"] = np.zeros(N); R["meas_q"] = np.zeros((N, NJ), np.float32); R["t_pub_ns"] = np.zeros(N, np.int64); R["goal_idx"] = np.zeros(N, np.int8)
if GC is not None:   # (demo) 1 = goal switched on arrival at this step
    R["goal_arr"] = np.zeros(N, np.int8)
REC_GE = bool(args.rec_goal_err) or args.out_lpf != "none"
if REC_GE:   # (filter)
    R["raw_q"] = np.zeros((N, NJ), np.float32); R["goal_pe"] = np.full(N, np.nan); R["goal_qe"] = np.full(N, np.nan)
if HUMAN_ON:   # (H1)
    for _k in ("h_age", "h_ms", "r_ms", "near_min", "near_ms"):
        R[_k] = np.full(N, np.nan)
    for _k in ("h_new", "near_sph", "near_shape", "near_seg"):
        R[_k] = np.full(N, -1, np.int16)
    if HRECL is not None:   # (H2 rec) saved in <out>_human.npz, not in _steps.npz
        R["h_relay_idx"] = np.full(N, -1, np.int64); R["h_pumped"] = np.full(N, -1, np.int64)
        R["near_sph_min"] = np.full((N, MEFF_SV), np.nan, np.float32); R["near_sph_seg"] = np.full((N, MEFF_SV), -1, np.int8)
        log(f"(H2 rec) human-state recording on -> {args.out}_human.npz")
    if args.human_field and args.human_near:   # (H2 field) largest field cost at the measured joints per step
        R["hf_val"] = np.full(N, np.nan, np.float32); R["hf_tot"] = np.full(N, np.nan, np.float32); R["hf_sph"] = np.full(N, -1, np.int16); R["hf_seg"] = np.full(N, -1, np.int16)
if args.warm_reset:   # (real_prep3) start from the measured state: the warm-up solves have moved the filter's previous command
    if args.js_source == "shm":
        read_shm()
    _q0 = LAST[0][0].copy(); _cs = SFILT.cmd_joint_state
    if _cs is not None:
        _cs.position.copy_(torch.as_tensor(_q0, dtype=_cs.position.dtype, device=_cs.position.device).reshape(_cs.position.shape))
        for _t in (_cs.velocity, _cs.acceleration) + ((_cs.jerk,) if _cs.jerk is not None else ()):
            _t.zero_()
    if args.acc_clip:
        CLIP["p1"] = _q0.copy(); CLIP["p2"] = _q0.copy()
    log(f"warm reset: solver state filter set to the measured joints (velocity 0) {np.round(_q0, 4).tolist()}")
LPREV = [None]; LGUARD = [0]; KPREV = [None]; KCAND = [None]; LREUSE = [0]   # (filter) --lim-ref cmd: last published first point, guard count
HGS = None   # (H2 gap) data-gap slowdown: current cap, base cap, counters
if args.human_gap_slow and HUMAN_ON and SPEED_LIM > 0:
    _hb_ = args.lim_v if (args.lim_ref == "cmd" and args.lim_v > 0) else SPEED_LIM
    HGS = dict(base=_hb_, low=min(args.hgs_v, _hb_), cap=_hb_, rej=0, old=0, blind=0, slowed=0, min_cap=_hb_)
    R["hgs_cap"] = np.zeros(N); R["hgs_blind"] = np.zeros(N, np.int8)
    log(f"(H2 gap) data-gap slowdown on: cap {_hb_} -> {HGS['low']} m/s while rejected or human age > {args.hgs_age} s, rate {args.hgs_rate} m/s per s")
elif args.human_gap_slow:
    log("(H2 gap) --human-gap-slow ignored (no --human shm or no sphere_speed_limit)")
OLPF = None
if args.out_lpf != "none":   # (filter) output smoothing, reset to the measured joints just before the first step
    from rp5_out_filter import OutFilter
    OLPF = OutFilter(args.out_lpf, f_hz=args.out_lpf_hz, fs=60.0, n=args.out_lpf_n, deg=args.out_lpf_deg, m=args.out_lpf_m)
    if args.js_source == "shm":
        read_shm()
    OLPF.reset(LAST[0][0].copy())
    log(f"(filter) output smoothing {OLPF.describe()}, feedback {args.out_lpf_feedback} (clip feedback {args.clip_feedback if args.acc_clip else 'n/a'})")
import signal
SIGREQ = [0.0]   # (2026-11-20) Ctrl+C: stop publishing at the next check, keep the records
signal.signal(signal.SIGINT, lambda _s, _f: SIGREQ.__setitem__(0, time.time()))
RIN_EVERY = 0 if args.rinert_every == "new" else int(args.rinert_every); RIN_LAST = [0]   # (OVERNIGHT_H1)
if HUMAN_ON and args.hsel != "off":
    R["hmask"] = np.ones((N, G1M.N_SHAPES), bool); R["hsel_n"] = np.full(N, -1, np.int16)
arrivals = 0; leg_start = 0; T0 = pc() + 0.05
SKIP = dict(slots=0, events=0, late_max=0.0); N_RUN = N   # (H2) --skip-late bookkeeping
R["skip_before"] = np.zeros(N, np.int32); R["late_start"] = np.zeros(N)
for k in range(N):
    sched = T0 + (k + SKIP["slots"]) * DT; now = pc(); KIN0 = [None]   # (OVERNIGHT_H1) limiter kinematics of this step
    if args.skip_late:   # (H2) late by more than one period at the start: drop the overdue slots, wait for the next slot
        _late = now - sched; SKIP["late_max"] = max(SKIP["late_max"], _late); R["late_start"][k] = _late
        if _late > DT:
            _n = int(_late / DT) + 1; SKIP["slots"] += _n; SKIP["events"] += 1; R["skip_before"][k] = _n
            sched = T0 + (k + SKIP["slots"]) * DT
        if k + SKIP["slots"] >= N:   # the wall length stays duration: the run ends at slot N
            N_RUN = k; break
    if sched > now:
        time.sleep(sched - now)
    a = pc(); HT = HUMAN_HOOK(k); _ha = pc() if HUMAN_ON else 0.0
    if args.js_source == "shm":
        read_shm()
    sage = time.time() - LAST[2]; cur = state_from_last(bool(args.age_comp)); age = a - LAST[1]
    if HUMAN_ON:   # (H1) robot inverse inertia at the measured joints of this step
        _hb = pc()
        if RIN_EVERY == 1 or (RIN_EVERY > 1 and k % RIN_EVERY == 0) or (RIN_EVERY == 0 and (HT["new"] or k - RIN_LAST[0] >= 4)):   # (OVERNIGHT_H1)
            human_robot_inertia(LAST[0][0]); RIN_LAST[0] = k
        R["r_ms"][k] = (pc() - _hb) * 1000; R["h_ms"][k] = (_ha - a) * 1000
        R["h_new"][k] = int(HT["new"]); R["h_age"][k] = MEFF_INP.age_val
        if HGS is not None:   # (H2 gap) target cap from the data state, moved at most rate x DT per step
            _ro = HGS_REJ[0] == 1; _ol = MEFF_INP.age_val > args.hgs_age; _bl = _ro or _ol
            _tg = HGS["low"] if _bl else HGS["base"]; _st = args.hgs_rate * DT
            HGS["cap"] = min(max(_tg, HGS["cap"] - _st), HGS["cap"] + _st)
            HGS["rej"] += int(_ro); HGS["old"] += int(_ol); HGS["blind"] += int(_bl); HGS["slowed"] += int(HGS["cap"] < HGS["base"] - 1e-9)
            HGS["min_cap"] = min(HGS["min_cap"], HGS["cap"]); R["hgs_cap"][k] = HGS["cap"]; R["hgs_blind"][k] = int(_bl)
        if HRECL is not None:   # (H2 rec) record in use at this step
            R["h_relay_idx"][k] = int(HREC["relay_msgs"]); R["h_pumped"][k] = int(HUMAN.last_pumped)
    b = pc()
    if SIGREQ[0]:
        STOP.update(reason="SIGINT", step=k, wall=time.time(), age_ms=age * 1000); break
    if STALE > 0 and age > STALE:   # (real_prep4) stop for good, nothing published from this step on
        STOP.update(reason="stale joint state at step start", step=k, wall=time.time(), age_ms=age * 1000); break
    torch.cuda.synchronize(); s0 = pc()
    res = mpc.step(cur, max_attempts=S.MAX_ATTEMPTS); torch.cuda.synchronize(); s1 = pc()
    feas = bool(res.metrics.feasible.all()); cmd = res.js_action.get_ordered_joint_state(mpc.joint_names)
    esc = False
    if not feas and args.infeas_escape:   # (night task 1) which term + self-collision escape (nothing computed when off)
        _sr = mpc.solver.safety_rollout; _st = res.metrics.state
        _b = float(_sr.bound_constraint.forward(_st.state_seq).max())
        _w = float(_sr.primitive_collision_constraint.forward(_st.robot_spheres, env_query_idx=None).max()) if _sr.primitive_collision_constraint.enabled else 0.0
        _sc = float(_sr.robot_self_collision_constraint.forward(_st.robot_spheres).max()) if _sr.robot_self_collision_constraint.enabled else 0.0
        _selfv = lambda qq: _sr.robot_self_collision_constraint.forward(mpc.rollout_fn.compute_kinematics(JointState.from_position(qq, joint_names=mpc.joint_names)).robot_spheres.reshape(qq.shape[0], 1, -1, 4).contiguous()).reshape(-1)
        _q0 = cur.position.reshape(1, -1).clone(); _sm = float(_selfv(_q0)[0]); ESCQ = None
        if _sm > 0:   # measured state inside the self-collision buffer: gradient step out
            _h = 1e-3; _E = torch.eye(NJ, dtype=_q0.dtype, device=_q0.device) * _h; _v = _selfv(torch.cat([_q0 + _E, _q0 - _E], 0)); _g = (_v[:NJ] - _v[NJ:]) / (2 * _h)
            if float(_g.norm()) > 0:
                _qn = _q0 - args.esc_step * _g.reshape(1, -1) / _g.norm()
                _wn = float(_sr.primitive_collision_constraint.forward(mpc.rollout_fn.compute_kinematics(JointState.from_position(_qn, joint_names=mpc.joint_names)).robot_spheres.reshape(1, 1, -1, 4).contiguous(), env_query_idx=None).max()) if _sr.primitive_collision_constraint.enabled else 0.0
                _qnn = _qn.reshape(-1).cpu().numpy(); _lm = float(np.minimum(_qnn - JLO, JHI - _qnn).min())
                if _wn == 0 and _lm > 0.05:
                    ESCQ = _qnn; esc = True
        R["inf_term"][k] = int(_b > 0) + 2 * int(_w > 0) + 4 * int(_sc > 0) + 8 * int(esc); R["inf_self"][k] = (_sc, _sm)
    jm = RosJT(); jm.joint_names = NAMES; sc = 1.0; vmx = float("nan")
    if esc:   # (night task 1) escape step from the self-collision buffer (no plan; 0.004 rad per step keeps the spheres near 0.2 m/s; acc clip below applies)
        qm = cur.position.reshape(-1).cpu().numpy(); p = RosJTP(); p.positions = [float(x) for x in ESCQ]; p.velocities = [float(x) for x in (ESCQ - qm) / DT]
        p.accelerations = [0.0] * NJ; p.time_from_start = RosDuration(sec=0, nanosec=0 if C6 else int(DT * 1e9)); jm.points.append(p)
        sq, sqd, sqdd = ESCQ, (ESCQ - qm) / DT, np.zeros(NJ); _PLAN = None
        if OLPF is not None and args.out_lpf_at == "pre":
            OLPF.reset(ESCQ)
    elif feas and torch.isfinite(cmd.position).all() and mpc.solver._init_act_seq is not None:
        a0 = cmd.acceleration.reshape(1, -1); ac = torch.cat([a0, mpc.solver._init_act_seq.reshape(-1, NJ)[:-1]], 0)
        v0 = cmd.velocity.reshape(1, -1) - a0 * DTF; p0 = cmd.position.reshape(1, -1) - cmd.velocity.reshape(1, -1) * DTF   # filtered start state (see step2b)
        hqd = v0 + torch.cumsum(ac * TRAJ_DT[:, None], 0); hq = p0 + torch.cumsum(hqd * TRAJ_DT[:, None], 0)
        HQ, HQD, HQDD = hq.cpu().numpy(), hqd.cpu().numpy(), ac.cpu().numpy()
        if args.rec_plan:   # (recplan 1124) the plan as solved (before the limiter / clip / filter change HQ[0]); numpy copy of the CPU horizon
            R["plan_q"][k] = HQ[:NPLAN]
        if OLPF is not None and args.out_lpf_at == "pre":   # (filter) smoothing before the limiter
            HQ[0], HQD[0], HQDD[0] = OLPF.step(np.asarray(HQ[0], float), plan_q=HQ if OLPF.needs_plan else None)
            hq = hq.clone(); hq[0] = torch.as_tensor(HQ[0], dtype=hq.dtype, device=hq.device)
        if SPEED_LIM > 0:   # hard limit on the first point, measured -> commanded, all robot spheres
            _k0 = mpc.rollout_fn.compute_kinematics(cur); k0 = _k0.robot_spheres.reshape(-1, 4)[:, :3].clone()
            if args.reuse_fk:   # (OVERNIGHT_H1) clones: the kinematics buffers are reused by the next call
                KIN0[0] = types.SimpleNamespace(ee_pos_seq=_k0.ee_pos_seq.clone(), ee_quat_seq=_k0.ee_quat_seq.clone(), robot_spheres=_k0.robot_spheres.clone())
            j1 = cur.clone(); j1.position = hq[0:1].clone()
            k1 = mpc.rollout_fn.compute_kinematics(j1).robot_spheres.reshape(-1, 4)[:, :3]
            vmx = float((k1 - k0).norm(dim=-1).max()) / DT if not (args.lim_ref == "cmd" and LPREV[0] is not None) else float("nan")   # (filter) cmd: one transfer below
            if args.lim_ref == "cmd" and LPREV[0] is not None:   # (filter) limit the commanded step from the previous published point (numpy)
                _K = torch.stack([k0, k1]).cpu().numpy().astype(np.float64); k0n, k1n = _K[0], _K[1]
                if args.lim_reuse and KPREV[0] is not None:
                    kpn = KPREV[0]; LREUSE[0] += 1
                else:
                    jp = cur.clone(); jp.position = torch.as_tensor(LPREV[0], dtype=hq.dtype, device=hq.device).reshape(1, -1)
                    kpn = mpc.rollout_fn.compute_kinematics(jp).robot_spheres.reshape(-1, 4)[:, :3].cpu().numpy().astype(np.float64)
                vmx = float(np.linalg.norm(k1n - kpn, axis=-1).max()) / DT; LV = args.lim_v if args.lim_v > 0 else SPEED_LIM
                if HGS is not None:   # (H2 gap)
                    LV = min(LV, HGS["cap"])
                if vmx > LV:
                    sc = LV / vmx; HQ[0] = LPREV[0] + sc * (HQ[0] - LPREV[0]); HQD[0] *= sc; HQDD[0] *= sc; k1n = kpn + sc * (k1n - kpn)
                KCAND[0] = (k1n, HQ[0].copy())
                if args.lim_guard > 0:   # guard against the measured joints (sphere positions interpolated linearly)
                    vg = float(np.linalg.norm(k1n - k0n, axis=-1).max()) / DT; GL = args.lim_guard * SPEED_LIM
                    if vg > GL:
                        sg = GL / vg; sc *= sg; qm = cur.position.reshape(-1).cpu().numpy(); HQ[0] = qm + sg * (HQ[0] - qm); HQD[0] *= sg; HQDD[0] *= sg; LGUARD[0] += 1; KCAND[0] = None
            elif vmx > (min(SPEED_LIM, HGS["cap"]) if HGS is not None else SPEED_LIM):
                sc = (min(SPEED_LIM, HGS["cap"]) if HGS is not None else SPEED_LIM) / vmx; qm = cur.position.reshape(-1).cpu().numpy(); HQ[0] = qm + sc * (HQ[0] - qm); HQD[0] *= sc; HQDD[0] *= sc
        for i in range(min(args.k, len(HT_NP))):
            p = RosJTP(); p.positions = [float(x) for x in HQ[i]]; p.velocities = [float(x) for x in HQD[i]]; p.accelerations = [float(x) for x in HQDD[i]]
            ns = int(round(float(HT_NP[i] - (HT_NP[0] if C6 else 0.0)) * 1e9)); p.time_from_start = RosDuration(sec=ns // 1000000000, nanosec=ns % 1000000000); jm.points.append(p)
        sq, sqd, sqdd = HQ[0], HQD[0], HQDD[0]; _PLAN = HQ   # (filter) _PLAN: horizon for --out-lpf blend
    else:
        qm = cur.position.reshape(-1).cpu().numpy(); p = RosJTP(); p.positions = [float(x) for x in qm]; p.velocities = [0.0] * NJ; p.accelerations = [0.0] * NJ
        p.time_from_start = RosDuration(sec=0, nanosec=0 if C6 else int(DT * 1e9)); jm.points.append(p); sq, sqd, sqdd = qm, np.zeros(NJ), np.zeros(NJ); _PLAN = None
        if OLPF is not None and args.out_lpf_at == "pre":   # (filter)
            OLPF.reset(qm)
    if REC_GE:
        R["raw_q"][k] = sq
    if OLPF is not None and args.out_lpf_at == "post":   # (filter) smoothing of the first point, after the speed limiter, before the acceleration clip
        sq, sqd, sqdd = OLPF.step(np.asarray(sq, float), plan_q=_PLAN if OLPF.needs_plan else None)
        _p0 = jm.points[0]; _p0.positions = [float(x) for x in sq]; _p0.velocities = [float(x) for x in sqd]; _p0.accelerations = [float(x) for x in sqdd]
        if args.out_lpf_feedback and not (args.acc_clip and args.clip_feedback) and SFILT.cmd_joint_state is not None:
            _cs = SFILT.cmd_joint_state
            _cs.position.copy_(torch.as_tensor(sq, dtype=_cs.position.dtype, device=_cs.position.device).reshape(_cs.position.shape))
            _cs.velocity.copy_(torch.as_tensor(sqd, dtype=_cs.velocity.dtype, device=_cs.velocity.device).reshape(_cs.velocity.shape))
            _cs.acceleration.copy_(torch.as_tensor(sqdd, dtype=_cs.acceleration.dtype, device=_cs.acceleration.device).reshape(_cs.acceleration.shape))
    if args.acc_clip:   # (real_prep3) last stage before publishing
        sq = np.asarray(sq, float); sqd = np.asarray(sqd, float); sqdd = np.asarray(sqdd, float)
        R["acc_pre"][k] = ((sq - CLIP["p1"]) / DT - (CLIP["p1"] - CLIP["p2"]) / DT) / DT
        sq, sqd, sqdd, dq, chg = acc_clip(sq, sqd, sqdd)
        if chg:
            p0 = jm.points[0]; p0.positions = [float(x) for x in sq]; p0.velocities = [float(x) for x in sqd]; p0.accelerations = [float(x) for x in sqdd]
            R["clip_dq"][k] = dq; R["clipped"][k] = 1
        if args.clip_feedback and SFILT.cmd_joint_state is not None:   # the solver's "previous command" = what was sent
            _cs = SFILT.cmd_joint_state
            _cs.position.copy_(torch.as_tensor(sq, dtype=_cs.position.dtype, device=_cs.position.device).reshape(_cs.position.shape))
            _cs.velocity.copy_(torch.as_tensor(sqd, dtype=_cs.velocity.dtype, device=_cs.velocity.device).reshape(_cs.velocity.shape))
            _cs.acceleration.copy_(torch.as_tensor(sqdd, dtype=_cs.acceleration.dtype, device=_cs.acceleration.device).reshape(_cs.acceleration.shape))
        CLIP["p2"] = CLIP["p1"]; CLIP["p1"] = np.asarray(sq, float).copy()
    if args.lim_ref == "cmd":   # (filter)
        LPREV[0] = np.asarray(sq, float).copy()
        KPREV[0] = KCAND[0][0] if (KCAND[0] is not None and np.allclose(LPREV[0], KCAND[0][1], rtol=0, atol=1e-9)) else None; KCAND[0] = None
    if STALE > 0:   # (real_prep4) newest receive time just before publishing
        _pa = pc() - (SB[17] if (SHM is not None and len(SB) > 17) else LAST[1]); R["pub_age"][k] = _pa
        if _pa > STALE:
            STOP.update(reason="stale joint state before publish", step=k, wall=time.time(), age_ms=_pa * 1000); break
    if SIGREQ[0]:
        STOP.update(reason="SIGINT", step=k, wall=time.time(), age_ms=(pc() - LAST[1]) * 1000); break
    if C6:
        _ns = time.time_ns(); jm.header.stamp.sec = _ns // 1000000000; jm.header.stamp.nanosec = _ns % 1000000000
    pub.publish(jm); tp = pc(); tpn = time.time_ns()
    # goal switching on the measured state
    kin = KIN0[0] if (args.reuse_fk and KIN0[0] is not None) else mpc.rollout_fn.compute_kinematics(cur); ee = kin.ee_pos_seq.reshape(-1, 3)[0]; qq = kin.ee_quat_seq.reshape(-1, 4)[0]
    if HUMAN_ON and args.hsel != "off":   # (OVERNIGHT_H1) robot spheres at the measured joints for the next selection; mask of this step
        HSEL["sph"] = kin.robot_spheres.reshape(-1, 4)[:MEFF_SV].cpu().numpy().astype(np.float64); R["hmask"][k] = HSEL["mask"]; R["hsel_n"][k] = HSEL["n"]
        if args.hsel_every == "step" and HSEL["h"] is not None:
            hsel_apply(HSEL["h"])
    if HUMAN_ON and args.human_near and k % args.human_near == 0:   # (H1) minimum gap robot sphere - human shape at the measured joints (surfaces), after publishing
        _n0 = pc(); _qa = MEFF.query_all(kin.robot_spheres.reshape(-1, 4)[:MEFF_SV]); _dd = _qa["dist"]; _j = int(torch.argmin(_dd))
        if "hf_val" in R:   # (H2 field) largest field cost over the active (Top-K) shapes, one small transfer
            _fa = MEFF.field_all(_qa, HSEL["n"] if args.hsel != "off" else G1M.N_SHAPES); _fj = int(torch.argmax(_fa["field"]))
            _fv = torch.stack([_fa["field"].reshape(-1)[_fj], _fa["total"].reshape(-1)[_fj], _qa["seg"].reshape(-1)[_fj].to(_fa["field"].dtype)]).cpu().numpy()
            R["hf_val"][k] = _fv[0]; R["hf_tot"][k] = _fv[1]; R["hf_sph"][k] = _fj // _dd.shape[1]; R["hf_seg"][k] = int(_fv[2])
        R["near_min"][k] = float(_dd.reshape(-1)[_j]); R["near_sph"][k] = _j // _dd.shape[1]; R["near_shape"][k] = _j % _dd.shape[1]
        R["near_seg"][k] = int(MEFF_INP.seg[_j % _dd.shape[1]]); R["near_ms"][k] = (pc() - _n0) * 1000
        if HRECL is not None:   # (H2 rec) nearest active shape per robot sphere: distance and segment, one small transfer
            _dm = torch.where(MEFF_INP.rad[None] > 0, _dd, torch.full_like(_dd, float("inf"))); _v, _i = _dm.min(1)
            _vs = torch.stack([_v, MEFF_INP.seg[_i].to(_v.dtype)]).cpu().numpy(); R["near_sph_min"][k] = _vs[0]; R["near_sph_seg"][k] = _vs[1]
    pe = float(torch.linalg.norm(ee - gb.goal_pose.position.reshape(-1, 3)[0])); qe = float(2 * torch.acos(torch.dot(qq, gb.goal_pose.quaternion.reshape(-1, 4)[0]).abs().clamp(0, 1)))
    if REC_GE:   # (filter) error to the goal of this step (before a switch)
        R["goal_pe"][k] = pe; R["goal_qe"][k] = qe
    if GC is not None:   # (demo) waypoint cycle
        _tp, _tq = GC["tol"][goal_idx]; _ok = pe <= _tp and qe <= _tq
        if _ok or (k - leg_start) >= LEG_STEPS:
            arrivals += int(_ok); GC["arr" if _ok else "timeouts"][goal_idx] += 1; R["goal_arr"][k] = int(_ok)
            if goal_idx == 0:
                GC["lap_steps"].append(k)
            goal_idx = (goal_idx + 1) % GC["n"]; gb.goal_pose.copy_(Pose.from_list(S.GOALS_WXYZ[goal_idx], ta)); mpc.update_goal(gb); leg_start = k
    elif (pe <= APT and qe <= S.ROT_TOL_RAD) or (k - leg_start) >= LEG_STEPS:
        arrivals += int(pe <= APT and qe <= S.ROT_TOL_RAD); goal_idx = 1 - goal_idx; gb.goal_pose.copy_(Pose.from_list(S.GOALS_WXYZ[goal_idx], ta)); mpc.update_goal(gb); leg_start = k
    e = pc()
    for key, val in (("t_sched", sched), ("t_start", a), ("t_state", b), ("t_solve0", s0), ("t_solve1", s1), ("t_pub", tp), ("t_end", e), ("state_age", age), ("stamp_age", sage), ("lim_scale", sc), ("sphere_v_before", vmx)):
        R[key][k] = val
    R["feasible"][k] = int(feas); R["sent_q"][k] = sq; R["sent_qd"][k] = sqd; R["sent_qdd"][k] = sqdd; R["meas_q"][k] = cur.position.reshape(-1).cpu().numpy(); R["t_pub_ns"][k] = tpn; R["goal_idx"][k] = goal_idx
if args.skip_late and not STOP and N_RUN < N:   # (H2) fewer steps than slots
    N = N_RUN; R = {kk: vv[:N] for kk, vv in R.items()}
if STOP:   # (real_prep4) keep only the published steps
    log(f"STOP: {STOP['reason']}, step {STOP['step']}, age {STOP['age_ms']:.1f} ms, wall {STOP['wall']:.6f}, last published step {STOP['step'] - 1}; nothing more is sent")
    N = STOP["step"]; R = {kk: vv[:N] for kk, vv in R.items()}
ex.shutdown(); node.destroy_node(); rclpy.shutdown()
if SHM is not None:
    del SB; SHM.close()
if HSHM is not None:   # (H1)
    del HB; HSHM.close()
for kk in [x for x in R if x.startswith("t_") and x != "t_pub_ns"]:
    R[kk] = R[kk] - T0
HREC_R = {kk: R.pop(kk) for kk in ("h_relay_idx", "h_pumped", "near_sph_min", "near_sph_seg") if kk in R}; HREC_INFO = None   # (H2 rec)
if HRECL is not None:   # (H2 rec) <out>_human.npz
    _SEGN = ["pelvis", "left_thigh", "right_thigh", "abdomen", "left_calf", "right_calf", "left_foot", "right_foot", "chest", "neck", "left_shoulder",
             "right_shoulder", "head", "left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm", "left_hand", "right_hand"]
    _M = np.array([x[2] for x in HRECL], np.float64).reshape(-1, HLEN); _g = lambda kk: _M[:, HOFF[kk][0]:HOFF[kk][0] + HOFF[kk][1]]
    _p0 = _g("p0").reshape(-1, 24, 3).astype(np.float32); _p1 = _g("p1").reshape(-1, 24, 3).astype(np.float32)
    _sm = HREC_R["near_sph_min"].astype(np.float64); _sm[~np.isfinite(_sm)] = np.inf; _ss = HREC_R["near_sph_seg"]
    _links = list(dict.fromkeys(RI.sphere_link)); _li = np.array([_links.index(x) for x in RI.sphere_link])
    _lm = np.full((len(_sm), len(_links)), np.nan, np.float32); _lseg = np.full((len(_sm), len(_links)), -1, np.int8)
    for _l in range(len(_links)):
        _c = np.nonzero(_li == _l)[0]; _a = _sm[:, _c].argmin(1) if len(_sm) else np.zeros(0, int); _mn = _sm[np.arange(len(_sm)), _c[_a]] if len(_sm) else np.zeros(0)
        _ok = np.isfinite(_mn); _lm[_ok, _l] = _mn[_ok]; _lseg[_ok, _l] = _ss[np.arange(len(_sm)), _c[_a]][_ok]
    _sm32 = HREC_R["near_sph_min"].copy(); _sm32[~np.isfinite(_sm32)] = np.nan; _ss2 = _ss.copy(); _ss2[~np.isfinite(_sm32)] = -1
    _HD = dict(msg_step=np.array([x[0] for x in HRECL], np.int32), msg_pick_wall=np.array([x[1] for x in HRECL], np.float64),
               msg_relay_idx=_M[:, 7].astype(np.int64), msg_stamp_ns=_M[:, 1].astype(np.int64) * 1000000000 + _M[:, 2].astype(np.int64),
               msg_recv_wall=_M[:, 3], msg_recv_mono=_M[:, 6], msg_input_count=_M[:, 4].astype(np.int64), msg_compute_ms=_M[:, 5].astype(np.float32),
               msg_com=_g("com").reshape(-1, 19, 3).astype(np.float32), msg_vel=_g("vel").reshape(-1, 19, 3).astype(np.float32),
               msg_inv=_g("inv").reshape(-1, 19, 3, 3).astype(np.float32), msg_p0=_p0, msg_p1=_p1, msg_center=(0.5 * (_p0 + _p1)).astype(np.float32),
               msg_rad=_g("rad").astype(np.float32), msg_seg=np.rint(_g("seg")).astype(np.int8), msg_type=np.rint(_g("type")).astype(np.uint8),
               step_t_pub_ns=R["t_pub_ns"], step_relay_idx=HREC_R["h_relay_idx"], step_pumped=HREC_R["h_pumped"], step_age_s=R["h_age"].astype(np.float32),
               step_new=R["h_new"].astype(np.int8), step_near_min=R["near_min"].astype(np.float32), sph_min=_sm32, sph_seg=_ss2, link_min=_lm, link_seg=_lseg,
               link_names=np.array(_links), sphere_link=np.array(RI.sphere_link), seg_names=np.array(_SEGN), frame_R=np.asarray(HUMAN.R, np.float64),
               frame_t=np.asarray(HUMAN.t, np.float64), frame_file=np.array(HFRAME), human_shm=np.array(args.human_shm), meas_q=R["meas_q"])
    if "hmask" in R:
        _HD.update(step_hmask=R["hmask"], step_hsel_n=R["hsel_n"])
    def _txt(p_):   # (H2 rec) file text, "" if not readable
        try:
            return open(os.path.expanduser(p_)).read() if p_ else ""
        except OSError:
            return ""
    _HD.update(msg_jf_rejected=np.array([x[3] if len(x) > 3 else 0 for x in HRECL], np.int8), jump_filter_on=np.array(int(JF is not None)),
               scene_file=np.array(os.path.abspath(os.path.expanduser(args.scene_json)) if args.scene_json else ""), scene_text=np.array(_txt(args.scene_json)),
               profile_file=np.array(os.path.abspath(os.path.expanduser(args.profile))), profile_text=np.array(_txt(args.profile)),
               camera_pose_file=np.array(os.path.abspath(os.path.expanduser(args.rec_camera_pose)) if args.rec_camera_pose else ""),
               camera_pose_text=np.array(_txt(args.rec_camera_pose)))
    if BEX is not None:   # (baseex)
        _HD.update(msg_bex_excluded=np.array(BEX_FL, np.int8), base_exclude_file=np.array(BEX.path), base_exclude_text=np.array(_txt(BEX.path)))
    np.savez(args.out + "_human.npz", **_HD)
    _ri = np.unique(_HD["msg_relay_idx"]); _ri = _ri[_ri > 0]
    HREC_INFO = dict(file=args.out + "_human.npz", bytes=os.path.getsize(args.out + "_human.npz"), messages=int(len(HRECL)),
                     messages_in_loop=int((_HD["msg_step"] >= 0).sum()), relay_idx_first=int(_ri[0]) if len(_ri) else None, relay_idx_last=int(_ri[-1]) if len(_ri) else None,
                     relay_records_not_seen=int(_ri[-1] - _ri[0] + 1 - len(_ri)) if len(_ri) else None)
    log(f"(H2 rec) {HREC_INFO}")
_HM = {k_: np.array(v_) for k_, v_ in HMETA.items()} if HUMAN_ON else {}   # (kgain) human cost meta
np.savez(args.out + "_steps.npz", **R, **({"plan_t": HT_NP[:NPLAN].astype(np.float32)} if args.rec_plan else {}), horizon_t=HT_NP, joint_names=np.array(NAMES), k=args.k, profile=args.profile, **_HM)
if N < 3:   # (real_prep4) stopped before 3 published steps: short summary only
    json.dump(dict(profile=args.profile, link=args.link, js_topic=args.js_topic, steps=N, stale_ms=args.stale_ms, stop=STOP or None, js_received=NRECV[0]), open(args.out + "_summary.json", "w"), indent=1)
    np.savez(args.out + "_steps.npz", **R, **_HM); log(f"only {N} steps published; short summary"); sys.exit(0)
solve = (R["t_solve1"] - R["t_solve0"]) * 1000; comp = (R["t_pub"] - R["t_start"]) * 1000; late = (R["t_pub"] - R["t_sched"]) * 1000; per = np.diff(R["t_pub_ns"]) / 1e6
qd2 = np.abs(np.diff(R["sent_q"].astype(float), 2, axis=0)) * 3600.0
S_ = dict(profile=args.profile, js_topic=args.js_topic, jt_topic=args.jt_topic, k=args.k, steps=N, duration_s=args.duration, wall_s=float((R["t_pub_ns"][-1] - R["t_pub_ns"][0]) / 1e9),
          msgs_per_s=float((N - 1) / ((R["t_pub_ns"][-1] - R["t_pub_ns"][0]) / 1e9)), period_ms=dict(mean=float(per.mean()), p99=float(np.percentile(per, 99)), max=float(per.max()), min=float(per.min())),
          solve_ms=dict(mean=float(solve.mean()), p95=float(np.percentile(solve, 95)), p99=float(np.percentile(solve, 99)), max=float(solve.max())),
          step_compute_ms=dict(mean=float(comp.mean()), p95=float(np.percentile(comp, 95)), p99=float(np.percentile(comp, 99)), max=float(comp.max())),
          overrun_steps=int((comp > DT * 1000).sum()), overrun_pct=float(100 * (comp > DT * 1000).mean()), deadline_miss=int((late > DT * 1000).sum()), late_ms_max=float(late.max()),
          sleep_lag_ms=dict(mean=float(((R["t_start"] - R["t_sched"]) * 1000).mean()), p99=float(np.percentile((R["t_start"] - R["t_sched"]) * 1000, 99)), max=float(((R["t_start"] - R["t_sched"]) * 1000).max())),
          state_age_ms=dict(mean=float(R["state_age"].mean() * 1000), p99=float(np.percentile(R["state_age"], 99) * 1000), max=float(R["state_age"].max() * 1000)),
          stamp_age_ms=dict(mean=float(R["stamp_age"].mean() * 1000), std=float(R["stamp_age"].std() * 1000), p99=float(np.percentile(R["stamp_age"], 99) * 1000), max=float(R["stamp_age"].max() * 1000)),
          js_source=args.js_source, age_comp=args.age_comp, js_qos=args.js_qos, js_received=NRECV[0], infeasible=int((R["feasible"] == 0).sum()), arrivals=arrivals,
          diffacc_over10_steps=int((qd2.max(1) > 10).sum()), diffacc_excess_max=float(max(qd2.max() - 10, 0)), diffacc_max_per_joint=qd2.max(0).tolist(),
          limiter_scaled_steps=int((R["lim_scale"] < 1).sum()), acc_clip=args.acc_clip, warm_reset=args.warm_reset, acc_max=args.acc_max, vel_frac=args.vel_frac, clip_feedback=args.clip_feedback,
          clipped_steps=int(R["clipped"].sum()), clip_dq_max_per_joint=np.abs(R["clip_dq"]).max(0).tolist(), warmup_ms=WARM, joint_limits=JLIM, sphere_speed_limit=SPEED_LIM, profile_keys_not_applied=NOT_APPLIED)
S_.update(link=args.link, js_topic=args.js_topic, stale_ms=args.stale_ms, stop=STOP or None,
          pub_age_ms=dict(mean=float(R["pub_age"].mean() * 1000), p99=float(np.percentile(R["pub_age"], 99) * 1000), max=float(R["pub_age"].max() * 1000)) if STALE > 0 else None,
          state_age_p50_ms=float(np.median(R["state_age"]) * 1000))
if args.infeas_escape:   # (night) infeasible terms and escapes
    S_["infeasible_terms"] = dict(bound=int(((R["inf_term"] & 1) > 0).sum()), world=int(((R["inf_term"] & 2) > 0).sum()), self_=int(((R["inf_term"] & 4) > 0).sum()),
                                  escaped=int(((R["inf_term"] & 8) > 0).sum()), esc_step=args.esc_step)
if GC is not None:   # (demo) waypoint cycle summary
    _ls = [i for i in GC["lap_steps"] if i < N]; _lt = [float((R["t_pub_ns"][j] - R["t_pub_ns"][i]) / 1e9) for i, j in zip(_ls[:-1], _ls[1:])]
    S_["goal_cycle"] = dict(n_goals=GC["n"], via=GC["via"], via_tol=args.via_tol, via_rot_tol=args.via_rot_tol, arrivals_per_goal=GC["arr"],
                            timeouts_per_goal=GC["timeouts"], laps=len(_ls), lap_s=_lt, lap_s_mean=float(np.mean(_lt)) if _lt else None)
if args.arrive_pos_tol is not None:   # (arrtol) summary (only with the option)
    S_.update(arrive_pos_tol=APT)
if args.leg_timeout is not None:   # (cam20) summary (only with the option)
    S_.update(leg_timeout_steps=LEG_STEPS, leg_timeout_s=round(LEG_STEPS / 60.0, 4))
if HUMAN_ON:   # (H1) human summary
    _nm = R["near_min"]; _ok = np.isfinite(_nm) & (_nm < 100.0)   # >= 100 m: human cost off after a tracking loss (shapes moved away)
    _jm = int(np.nanargmin(np.where(_ok, _nm, np.nan))) if _ok.any() else -1
    _segn = ["pelvis", "left_thigh", "right_thigh", "abdomen", "left_calf", "right_calf", "left_foot", "right_foot", "chest", "neck", "left_shoulder",
             "right_shoulder", "head", "left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm", "left_hand", "right_hand"]
    _pct = lambda x, q: float(np.nanpercentile(x, q)) if np.isfinite(x).any() else None
    S_["human"] = dict(HINFO, relay_messages=int(HREC["relay_msgs"]), pickups=HREC["pickups"], pumped=int(HUMAN.last_pumped), new_steps=int((R["h_new"] == 1).sum()),
                       age_ms=dict(mean=float(np.nanmean(R["h_age"]) * 1000), p99=_pct(R["h_age"], 99) * 1000, max=float(np.nanmax(R["h_age"]) * 1000)),
                       hook_ms=dict(p99=_pct(R["h_ms"], 99), max=float(np.nanmax(R["h_ms"]))), robot_inertia_ms=dict(p99=_pct(R["r_ms"], 99), max=float(np.nanmax(R["r_ms"]))),
                       near=dict(on=args.human_near, min_m=float(_nm[_jm]) if _jm >= 0 else None, step=_jm, t_s=float(R["t_pub"][_jm]) if _jm >= 0 else None,
                                 sphere=int(R["near_sph"][_jm]) if _jm >= 0 else None, sphere_link=RI.sphere_link[int(R["near_sph"][_jm])] if _jm >= 0 else None,
                                 segment=_segn[int(R["near_seg"][_jm])] if _jm >= 0 else None, steps_below_0p10=int((_nm[_ok] < 0.10).sum()), steps_below_0p05=int((_nm[_ok] < 0.05).sum()),
                                 query_ms_p99=_pct(R["near_ms"], 99)) if args.human_near else None,
                       loss_events=[[int(a_), int(b_), float(c_)] for a_, b_, c_ in HUMAN.loss_events], final=HUMAN.final_check())
    if HREC_INFO is not None:   # (H2 rec)
        S_["human"]["rec"] = HREC_INFO
    if args.human_field or HDM_INFO is not None:   # (H2 field)
        _hf = dict(on=args.human_field, W=args.hf_wt, depth_margin=HDM_INFO)
        if "hf_val" in R and np.isfinite(R["hf_val"]).any():
            _fv_ = np.nan_to_num(R["hf_val"].astype(np.float64), nan=-1.0); _fk = int(np.argmax(_fv_))
            _hf.update(max=float(_fv_[_fk]), max_step=_fk, max_t_s=float(R["t_pub"][_fk]), max_sphere=int(R["hf_sph"][_fk]), max_link=RI.sphere_link[int(R["hf_sph"][_fk])],
                       max_segment=_segn[int(R["hf_seg"][_fk])] if R["hf_seg"][_fk] >= 0 else None, mean=float(np.nanmean(R["hf_val"])),
                       steps_on=int((np.nan_to_num(R["hf_val"]) > 0).sum()), steps_inside_wall=int((np.nan_to_num(R["hf_val"]) >= args.hf_wt - 1e-6).sum()) if args.hf_wt > 0 else None)
        S_["human"]["field"] = _hf; log(f"(H2 field) {_hf}")
    if JF is not None:   # (H2 jf)
        _jt0 = R["t_pub_ns"][0] / 1e9 if N else 0.0
        S_["human"]["jump_filter"] = dict(JF.describe(), **(JF.counters() if hasattr(JF, "counters") else {}), frames_seen=JF.n_seen, rejected=JF.n_rejected,
                                          episodes=[[round(a_ - _jt0, 3), round(b_ - _jt0, 3), int(n_)] for a_, b_, n_ in JF.events if n_ > 0])
        log(f"(H2 jf) {S_['human']['jump_filter']}")
    if HUMAN_ON and HSHIFT is not None:   # (cam20) summary
        S_["human"]["shift"] = HSHIFT.tolist()
    if HUMAN_ON and HX:   # (robex 1124) summary
        S_["human"]["robot_exclude"] = HXD
    if BEX is not None:   # (baseex) summary + one line
        from rp5_base_exclude import episodes as _bex_ep
        _bt0 = R["t_pub_ns"][0] / 1e9 if N else 0.0; _bep = [[round(a_ - _bt0, 3), round(b_ - _bt0, 3), int(n_)] for a_, b_, n_ in _bex_ep(BEX_T)]
        S_["human"]["base_exclude"] = dict(BEX.describe(), records_seen=BEX_N["seen"], excluded=BEX_N["excluded"], dropped_no_previous=BEX_N["no_previous"],
                                           episodes_s=_bep, episodes_note="[first, last, records], input stamp - first publish, s")
        log(f"(baseex) human base exclude: {BEX_N['excluded']} of {BEX_N['seen']} records excluded ({BEX_N['no_previous']} with no earlier person), "
            f"episodes s: {' '.join('%.2f-%.2f(%d)' % tuple(e) for e in _bep) or 'none'}")
    if HGS is not None:   # (H2 gap)
        S_["human"]["gap_slow"] = dict(cap_mps=HGS["base"], low_mps=HGS["low"], age_s=args.hgs_age, rate=args.hgs_rate, rejected_s=round(HGS["rej"] * DT, 2),
                                       old_s=round(HGS["old"] * DT, 2), blind_s=round(HGS["blind"] * DT, 2), slowed_s=round(HGS["slowed"] * DT, 2), min_cap_mps=round(HGS["min_cap"], 3))
        log(f"(H2 gap) {S_['human']['gap_slow']}")
    if HVC is not None:   # (H2 cost)
        S_["human"]["vel_clean"] = dict({k: v for k, v in HVC.items() if k != "prev_rej"}, max_mps=args.hvc_max); log(f"(H2 cost) {S_['human']['vel_clean']}")
_pc = lambda x: dict(p50=float(np.percentile(x, 50)), p99=float(np.percentile(x, 99)), max=float(np.max(x))) if len(x) else None
_hh = R["h_ms"] if "h_ms" in R else np.zeros(N); _rr = R["r_ms"] if "r_ms" in R else np.zeros(N)
S_.update(h1n=dict(reuse_fk=args.reuse_fk, rinert_every=args.rinert_every, human_near=args.human_near, meff_noop=args.meff_noop, hsel=args.hsel, hsel_k=args.hsel_k,
                   hsel_r=args.hsel_r, hsel_every=args.hsel_every, hsel_n_mean=float(R["hsel_n"][R["hsel_n"] >= 0].mean()) if "hsel_n" in R and (R["hsel_n"] >= 0).any() else None))
S_.update(step_split_ms=dict(   # (OVERNIGHT_H1) per step: total = start -> publish; other = pre-solve rest + solve end -> publish; after_publish = publish -> step end
    total=_pc(comp), solve=_pc(solve), human=_pc(_hh), robot_inertia=_pc(_rr),
    other=_pc((R["t_state"] - R["t_start"]) * 1000 - _hh - _rr + (R["t_solve0"] - R["t_state"]) * 1000 + (R["t_pub"] - R["t_solve1"]) * 1000),
    after_publish=_pc((R["t_end"] - R["t_pub"]) * 1000)))
S_.update(skip_late=args.skip_late, skipped_slots=SKIP["slots"], skip_events=SKIP["events"], late_at_start_max_ms=SKIP["late_max"] * 1000 if args.skip_late else None,
          mppi_particles=int(SET.get("mppi_particles", 0) or 0) or 400, mppi_iters=int(SET.get("mppi_iters", 0) or 0) or 5, step_compute_p50_ms=float(np.median(comp)))   # (H2)
if REC_GE or args.mpc_override:   # (filter) summary
    def _hf(q):
        q = np.asarray(q, np.float64)
        if len(q) < 30:
            return None
        a_ = np.diff(q, 2, axis=0) * 3600.0; a_ = a_ - a_.mean(0); F_ = np.abs(np.fft.rfft(a_ * np.hanning(len(a_))[:, None], axis=0)) ** 2; f_ = np.fft.rfftfreq(len(a_), DT)
        return float(100 * F_[f_ > 15].sum() / F_[f_ > 0].sum())
    _OF = dict(lpf=OLPF.describe() if OLPF else None, lpf_feedback=args.out_lpf_feedback, clip_feedback=args.clip_feedback, mpc_override=MPC_OVR,
               hf15_sent_pct=_hf(R["sent_q"]))
    if REC_GE:
        _g = R["goal_idx"].astype(int); _sw = np.nonzero(np.diff(_g))[0] + 1; _arr = [int(i) for i in _sw if R["goal_pe"][i] <= APT and R["goal_qe"][i] <= S.ROT_TOL_RAD]
        _OF.update(hf15_raw_pct=_hf(R["raw_q"]), arrivals_idx=_arr, arrival_pe_mm=[float(R["goal_pe"][i] * 1000) for i in _arr],
                   lpf_offset_mm_max=float(np.abs(R["sent_q"] - R["raw_q"]).max()) if OLPF else 0.0,
                   sent_acc_max=float(np.abs(np.diff(R["sent_q"].astype(np.float64), 2, axis=0)).max() * 3600.0) if N > 2 else None)
    _OF.update(out_lpf_at=args.out_lpf_at, lim_ref=args.lim_ref, lim_v=args.lim_v, lim_guard=args.lim_guard, lim_guard_steps=LGUARD[0], lim_reuse_steps=LREUSE[0])
    S_["out_filter"] = _OF
json.dump(S_, open(args.out + "_summary.json", "w"), indent=1)
log(json.dumps({k: v for k, v in S_.items() if k not in ("warmup_ms", "joint_limits")}))
