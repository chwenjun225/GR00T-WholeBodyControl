"""Pure mapping, torque and frame conversions, testable without Kit or DDS."""
import numpy as np

def joint_groups(with_hands=True):
    # Matches G1SupplementalInfo's hardware order. Avoid importing robot_model:
    # its package initializer pulls Pinocchio into Isaac Sim's Python environment.
    leg = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    arm = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
           "wrist_roll", "wrist_pitch", "wrist_yaw")
    body = [f"{side}_{name}_joint" for side in ("left", "right") for name in leg]
    body += [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    body += [f"{side}_{name}_joint" for side in ("left", "right") for name in arm]
    groups = {"body": tuple(body)}
    if with_hands:
        fingers = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
        for side in ("left", "right"):
            groups[side+"_hand"] = tuple(f"{side}_hand_{name}_joint" for name in fingers)
    return groups


def map_joints(dof_names, groups):
    if len(dof_names) != len(set(dof_names)):
        raise ValueError("Articulation contains duplicate DOF names")
    lookup = {name: i for i, name in enumerate(dof_names)}
    missing = [name for names in groups.values() for name in names if name not in lookup]
    if missing:
        raise ValueError(f"Missing G1 joints: {missing}; available: {list(dof_names)}")
    return {key: np.array([lookup[name] for name in names], dtype=np.int32)
            for key, names in groups.items()}


def command_torques(motors, q, dq, limits):
    """Unitree PR commands: feed-forward plus explicit PD, clipped in Nm."""
    if len(motors) < len(q):
        raise ValueError("Motor command is shorter than the controlled joint group")
    fields = np.array([[m.q, m.dq, m.kp, m.kd, m.tau] for m in motors[:len(q)]],
                      dtype=np.float64)
    if not np.isfinite(fields).all() or not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError("Non-finite joint state or motor command")
    if np.any(fields[:, 2:4] < 0):
        raise ValueError("Negative PD gains")
    # Unitree sentinel targets disable the respective PD term, not the feed-forward torque.
    pos_error = np.where(np.abs(fields[:, 0]) >= 1e9, 0, fields[:, 0] - q)
    vel_error = np.where(np.abs(fields[:, 1]) >= 16000, 0, fields[:, 1] - dq)
    torque = fields[:, 4] + fields[:, 2] * pos_error + fields[:, 3] * vel_error
    if not np.isfinite(torque).all():
        raise ValueError("Non-finite computed torque")
    return np.clip(torque, -limits, limits)


def world_to_body(quaternion_wxyz, vector):
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-8:
        raise ValueError("Invalid IMU quaternion")
    q = q / norm
    w, x, y, z = q
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return rotation.T @ np.asarray(vector)
