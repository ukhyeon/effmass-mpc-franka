# HumanInertiaState (goal1 1-2b). Frame: human_dynamics_node world (same as hrc HumanDynamicsState). Arrays row-major, segment order = segment_names (19).
std_msgs/Header header                 # stamp = time computation finished, frame_id = "world"
uint64 input_count                     # count of HumanPose messages received by this node (1-based)
builtin_interfaces/Time input_stamp    # header.stamp of the HumanPose used
string input_frame_id                  # header.frame_id of the HumanPose used
float32 compute_ms                     # compute time for this message
float32 velocity_dt                    # input stamp difference used for velocity [s] (0 when invalid)
bool velocity_valid                    # false for first message or invalid dt
string[] segment_names                 # 19
float32[] mass                         # 19 [kg]
float32[] com_position                 # 19*3 [m]
float32[] com_velocity                 # 19*3 [m/s]
float32[] rotation                     # 19*9 segment world rotation
float32[] joint_proximal               # 19*3 segment proximal joint
float32[] joint_distal                 # 19*3 segment distal point (child joint / tip)
float32[] inv_inertia                  # 19*9 task-space inverse inertia J_p Minv_contact J_p^T [1/kg] (structural locks + foot contact, no ROM direction locks)
uint8 SHAPE_SPHERE=0
uint8 SHAPE_CAPSULE=1
uint8 N_SHAPES=24                      # (1-2d) fixed shape count: torso 3 segs x2, hands 2 x2 (palm-fingers, thumb), others 14 x1 (head = capsule)
uint8[] shape_type
uint8[] shape_segment                  # owning segment index (0..18)
float32[] shape_p0                     # n*3 capsule end 0 / sphere center
float32[] shape_p1                     # n*3 capsule end 1 / sphere center
float32[] shape_radius                 # n
