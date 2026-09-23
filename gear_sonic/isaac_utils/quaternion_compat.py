"""Quaternion compatibility for SONIC on Isaac Lab 3.x.

SONIC's motion, policy, and checkpoint code uses scalar-first ``wxyz``
quaternions. Isaac Lab 3.x exposes simulation tensors and math helpers in
scalar-last ``xyzw`` order. This module keeps SONIC's internal convention and
converts only at Isaac Lab math/API boundaries.
"""

from isaaclab.utils import math as _math


def to_sim_quat(quat):
    """Convert a SONIC ``wxyz`` quaternion to Isaac Lab 3 ``xyzw``."""
    return quat[..., [1, 2, 3, 0]]


def from_sim_quat(quat):
    """Convert an Isaac Lab 3 ``xyzw`` quaternion to SONIC ``wxyz``."""
    return quat[..., [3, 0, 1, 2]]


def matrix_from_quat(quat):
    return _math.matrix_from_quat(to_sim_quat(quat))


def quat_apply(quat, vec):
    return _math.quat_apply(to_sim_quat(quat), vec)


def quat_apply_inverse(quat, vec):
    return _math.quat_apply_inverse(to_sim_quat(quat), vec)


def quat_apply_yaw(quat, vec):
    return _math.quat_apply_yaw(to_sim_quat(quat), vec)


def quat_conjugate(quat):
    return from_sim_quat(_math.quat_conjugate(to_sim_quat(quat)))


def quat_inv(quat):
    return from_sim_quat(_math.quat_inv(to_sim_quat(quat)))


def quat_mul(quat1, quat2):
    return from_sim_quat(_math.quat_mul(to_sim_quat(quat1), to_sim_quat(quat2)))


def quat_error_magnitude(quat1, quat2):
    return _math.quat_error_magnitude(to_sim_quat(quat1), to_sim_quat(quat2))


def quat_from_euler_xyz(roll, pitch, yaw):
    return from_sim_quat(_math.quat_from_euler_xyz(roll, pitch, yaw))


def axis_angle_from_quat(quat, eps=1.0e-6):
    return _math.axis_angle_from_quat(to_sim_quat(quat), eps)


def subtract_frame_transforms(t01, q01, t02=None, q02=None):
    pos, quat = _math.subtract_frame_transforms(
        t01, to_sim_quat(q01), t02, None if q02 is None else to_sim_quat(q02)
    )
    return pos, from_sim_quat(quat)


sample_uniform = _math.sample_uniform
