# GR1T2 robot definition for SONIC / Isaac Lab

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

GR1T2_MUJOCO_JOINTS = [
    "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "head_pitch_joint", "head_roll_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
    "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
    "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
]

GR1T2_ISAACLAB_JOINTS = [
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_yaw_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_roll_joint",
    "left_knee_pitch_joint", "right_knee_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "head_roll_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "head_pitch_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "head_yaw_joint", "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
]

GR1T2_ISAACLAB_BODIES = [
    "base_link",
    "left_thigh_roll_link", "right_thigh_roll_link", "waist_yaw_link",
    "left_thigh_yaw_link", "right_thigh_yaw_link", "waist_pitch_link",
    "left_thigh_pitch_link", "right_thigh_pitch_link", "waist_roll_link",
    "left_shank_pitch_link", "right_shank_pitch_link", "head_roll_link",
    "left_upper_arm_pitch_link", "right_upper_arm_pitch_link",
    "left_foot_pitch_link", "right_foot_pitch_link", "head_pitch_link",
    "left_upper_arm_roll_link", "right_upper_arm_roll_link",
    "left_foot_roll_link", "right_foot_roll_link", "head_yaw_link",
    "left_upper_arm_yaw_link", "right_upper_arm_yaw_link",
    "left_lower_arm_pitch_link", "right_lower_arm_pitch_link",
    "left_hand_yaw_link", "right_hand_yaw_link",
    "left_hand_roll_link", "right_hand_roll_link",
    "left_hand_pitch_link", "right_hand_pitch_link",
]

GR1T2_ISAACLAB_TO_MUJOCO_DOF = [
    0, 3, 6, 9, 11, 16, 1, 4, 7, 10, 12, 17, 2, 5, 8, 20,
    15, 23, 13, 18, 21, 24, 26, 28, 30, 14, 19, 22, 25, 27, 29, 31,
]
GR1T2_MUJOCO_TO_ISAACLAB_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 4, 10, 18, 25, 16,
    5, 11, 19, 26, 15, 20, 27, 17, 21, 28, 22, 29, 23, 30, 24, 31,
]
GR1T2_ISAACLAB_TO_MUJOCO_BODY = [0] + [i + 1 for i in GR1T2_ISAACLAB_TO_MUJOCO_DOF]
GR1T2_MUJOCO_TO_ISAACLAB_BODY = [0] + [i + 1 for i in GR1T2_MUJOCO_TO_ISAACLAB_DOF]

GR1T2_ISAACLAB_TO_MUJOCO_MAPPING = {
    # Kept under the historical key used by TrackingCommand; this list is in
    # fact the Isaac Lab rigid-body order, including the root body.
    "isaaclab_joints": GR1T2_ISAACLAB_BODIES,
    "isaaclab_dof_joints": GR1T2_ISAACLAB_JOINTS,
    "mujoco_joints": GR1T2_MUJOCO_JOINTS,
    "isaaclab_to_mujoco_dof": GR1T2_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": GR1T2_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": GR1T2_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": GR1T2_MUJOCO_TO_ISAACLAB_BODY,
}

assert len(GR1T2_MUJOCO_JOINTS) == 32
assert len(GR1T2_ISAACLAB_JOINTS) == 32
assert len(GR1T2_ISAACLAB_BODIES) == 33
assert set(GR1T2_MUJOCO_JOINTS) == set(GR1T2_ISAACLAB_JOINTS)
assert sorted(GR1T2_ISAACLAB_TO_MUJOCO_DOF) == list(range(32))
assert sorted(GR1T2_MUJOCO_TO_ISAACLAB_DOF) == list(range(32))

GR1T2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/gr1t2/gr1t2_fourier_hand_6dof.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0, damping=0.0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.98),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_pitch_joint": 0.40,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint"],
            effort_limit_sim=60.0, stiffness=57.0, damping=5.7,
        ),
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint"],
            effort_limit_sim=45.0, stiffness=43.0, damping=4.3,
        ),
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint"],
            effort_limit_sim=130.0, stiffness=114.0, damping=11.4,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_pitch_joint"],
            effort_limit_sim=130.0, stiffness=114.0, damping=11.4,
        ),
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit_sim=16.0, stiffness=15.3, damping=1.5,
        ),
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit_sim=16.0, stiffness=15.3, damping=1.5,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim=60.0, stiffness=40.0, damping=4.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_.*_joint"],
            effort_limit_sim=20.0, stiffness=15.0, damping=1.5,
        ),
        "shoulders_elbows": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_pitch_joint"],
            effort_limit_sim=60.0, stiffness=40.0, damping=4.0,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*_joint"],
            effort_limit_sim=20.0, stiffness=15.0, damping=1.5,
        ),
    },
)

GR1T2_ACTION_SCALE = {}
for actuator in GR1T2_CFG.actuators.values():
    effort = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    names = actuator.joint_names_expr
    if not isinstance(effort, dict):
        effort = dict.fromkeys(names, effort)
    if not isinstance(stiffness, dict):
        stiffness = dict.fromkeys(names, stiffness)
    for name in names:
        if name in effort and name in stiffness and stiffness[name]:
            GR1T2_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
