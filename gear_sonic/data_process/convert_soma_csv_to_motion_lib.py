#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Convert SOMA retargeter CSV/PKL data to motion_lib format for SONIC training.

SOMA retargeter outputs GR1T2 32-DOF motion data as CSV files (joint_pos.csv,
body_pos.csv, body_quat.csv) or as a joblib PKL with the same fields. This
script converts that data into the motion_lib PKL format expected by SONIC
training (root_trans_offset, pose_aa, dof, root_rot, fps).

Supports five input modes:
  1. Single motion directory with CSVs (joint_pos.csv, body_pos.csv, body_quat.csv)
  2. Parent directory containing multiple motion subdirectories
  3. Deploy PKL file (joblib dict with joint_pos, body_pos_w, body_quat_w per sequence)
  4. Directory of flat Bones-SEED CSVs (single CSV per motion, degrees+cm)
  5. Parent directory of session dirs containing Bones-SEED CSVs

Usage:
    # Single CSV directory
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input data/soma_retarget/tired_squat_003__A360 \
        --output data/soma_test.pkl --fps 50

    # Batch: parent dir with multiple motion subdirs
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input data/soma_retarget/all_demo_4seqs \
        --output data/soma_demo_4seqs.pkl --fps 50

    # Deploy PKL file
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input data/soma_retarget/bones_test.pkl \
        --output data/soma_bones_test.pkl --fps 50

    # Bones-SEED: directory of flat CSVs (single session)
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input /path/to/bones_SEED/gr1t2/csv/210531 \
        --output data/bones_seed_210531.pkl --fps 50

    # Bones-SEED: all sessions (parent dir)
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input /path/to/bones_SEED/gr1t2/csv \
        --output data/bones_seed_all.pkl --fps 50
"""

import argparse
import os
from pathlib import Path
import sys
import warnings
import xml.etree.ElementTree as ETree

import joblib
import numpy as np
from scipy.spatial import transform

_CIBO_ROOT = Path(__file__).resolve().parents[4]
_GR1T2_ASSET_DIR = _CIBO_ROOT / "contents/assets/robots/humanoid/gr1t2"
GR1T2_MJCF_PATH = _GR1T2_ASSET_DIR / "mjcf/gr1t2_no_fingers.xml"
GR1T2_URDF_PATH = _GR1T2_ASSET_DIR / "urdf/gr1t2.urdf"

# Newton emits ``joint_q[7:]`` in this MJCF tree/coordinate order. Keep this
# aligned with envs/manager_env/robots/gr1t2.py and validate it against the XML
# below so a mislabeled CSV cannot silently corrupt the motion again.
BODY_JOINTS_MUJOCO = [
    "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
    "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
    "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "head_roll_joint", "head_pitch_joint", "head_yaw_joint",
]

# SOMA CSVs produced before the GR1T2 schema fix contain Newton's correct
# positional joint_q values under these incorrect column labels. Accept this
# exact legacy signature and read its 32 joint columns positionally.
LEGACY_BODY_JOINTS_CSV = [
    *BODY_JOINTS_MUJOCO[:15],
    "head_pitch_joint", "head_roll_joint", "head_yaw_joint",
    *BODY_JOINTS_MUJOCO[15:29],
]

_POLICY_JOINT_SET = set(BODY_JOINTS_MUJOCO)
BODY_JOINTS_ISAACLAB = [
    joint.get("name")
    for joint in ETree.parse(GR1T2_URDF_PATH).getroot().findall("joint")
    if joint.get("name") in _POLICY_JOINT_SET
]


def _load_gr1t2_mjcf_contract():
    """Read the GR1T2 worldbody contract without loading stale finger actuators."""
    root = ETree.parse(GR1T2_MJCF_PATH).getroot()
    xml_root_body = root.find("worldbody/body")
    if xml_root_body is None:
        raise ValueError(f"GR1T2 MJCF has no worldbody root: {GR1T2_MJCF_PATH}")

    body_names = []
    body_to_joint = {}
    joint_axis = {}
    joint_range = {}
    xml_joint_order = []

    def visit(body):
        body_index = len(body_names)
        body_name = body.get("name")
        if not body_name:
            raise ValueError("Every GR1T2 MJCF body must have a name")
        body_names.append(body_name)
        for joint in body.findall("joint"):
            name = joint.get("name")
            if name == "floating_base" or joint.get("type") == "free":
                continue
            if name in joint_axis:
                raise ValueError(f"Duplicate GR1T2 MJCF joint name: {name}")
            xml_joint_order.append(name)
            body_to_joint[body_index] = name
            joint_axis[name] = np.fromstring(joint.get("axis", ""), sep=" ", dtype=np.float32)
            joint_range[name] = np.fromstring(
                joint.get("range", ""), sep=" ", dtype=np.float32
            )
        for child in body.findall("body"):
            visit(child)

    visit(xml_root_body)
    return body_names, body_to_joint, xml_joint_order, joint_axis, joint_range


(
    POSE_AA_BODY_NAMES,
    POSE_AA_BODY_TO_JOINT,
    XML_JOINTS,
    _JOINT_AXIS_BY_NAME,
    _JOINT_RANGE_BY_NAME,
) = _load_gr1t2_mjcf_contract()

NUM_DOF = len(BODY_JOINTS_MUJOCO)
NUM_BODIES = len(POSE_AA_BODY_NAMES)
MJ_TO_IL = np.array(
    [BODY_JOINTS_ISAACLAB.index(name) for name in BODY_JOINTS_MUJOCO], dtype=np.int32
)
MJ_TO_XML = np.array([XML_JOINTS.index(name) for name in BODY_JOINTS_MUJOCO], dtype=np.int32)
XML_TO_MJ = np.array([BODY_JOINTS_MUJOCO.index(name) for name in XML_JOINTS], dtype=np.int32)
DOF_AXIS = np.stack([_JOINT_AXIS_BY_NAME[name] for name in BODY_JOINTS_MUJOCO])
DOF_RANGE = np.stack([_JOINT_RANGE_BY_NAME[name] for name in BODY_JOINTS_MUJOCO])
BONES_CSV_JOINT_NAMES = [f"{name}_dof" for name in BODY_JOINTS_MUJOCO]
BONES_CSV_HEADER = [
    "Frame",
    "root_translateX", "root_translateY", "root_translateZ",
    "root_rotateX", "root_rotateY", "root_rotateZ",
    *BONES_CSV_JOINT_NAMES,
]
LEGACY_BONES_CSV_HEADER = [
    *BONES_CSV_HEADER[:7],
    *(f"{name}_dof" for name in LEGACY_BODY_JOINTS_CSV),
]

assert NUM_DOF == 32
assert len(BODY_JOINTS_ISAACLAB) == NUM_DOF
assert len(XML_JOINTS) == NUM_DOF
assert set(XML_JOINTS) == set(BODY_JOINTS_MUJOCO) == set(BODY_JOINTS_ISAACLAB)
assert XML_JOINTS == BODY_JOINTS_MUJOCO
assert sorted(MJ_TO_IL.tolist()) == list(range(NUM_DOF))
assert sorted(MJ_TO_XML.tolist()) == list(range(NUM_DOF))
assert sorted(XML_TO_MJ.tolist()) == list(range(NUM_DOF))
assert DOF_AXIS.shape == (NUM_DOF, 3)
assert DOF_RANGE.shape == (NUM_DOF, 2)
assert np.isfinite(DOF_AXIS).all() and np.all(np.linalg.norm(DOF_AXIS, axis=1) > 0)
assert len(BONES_CSV_HEADER) == 39
assert len(LEGACY_BONES_CSV_HEADER) == 39

_LEGACY_HEADER_WARNED = False


def load_bones_csv(csv_path: str) -> dict:
    """Load a single Bones-SEED flat CSV motion file.

    GR1T2 CSV format: Frame, six root fields, then 32 canonical joint DOFs.
    All angles in degrees, positions in centimeters.
    """
    import pandas as pd

    data = pd.read_csv(csv_path)
    actual_header = data.columns.tolist()
    if actual_header not in (BONES_CSV_HEADER, LEGACY_BONES_CSV_HEADER):
        raise ValueError(
            f"GR1T2 CSV header mismatch for {csv_path}: expected the current or "
            f"known legacy SOMA schema, got {actual_header}"
        )
    T = len(data)

    # Root position: cm → meters
    root_pos = (
        np.stack(
            [
                data["root_translateX"].values,  # noqa: PD011
                data["root_translateY"].values,  # noqa: PD011
                data["root_translateZ"].values,  # noqa: PD011
            ],
            axis=1,
        ).astype(np.float32)
        / 100.0
    )  # cm → m

    # Root rotation: Euler xyz (intrinsic) degrees → quaternion (xyzw scipy convention)
    # Reference: gear_sonic/data_process/process_bones_to_motionlib.py uses "xyz" (intrinsic)
    euler_deg = np.stack(
        [
            data["root_rotateX"].values,  # noqa: PD011
            data["root_rotateY"].values,  # noqa: PD011
            data["root_rotateZ"].values,  # noqa: PD011
        ],
        axis=1,
    ).astype(np.float64)
    root_quat_xyzw = (
        transform.Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float32)
    )
    # Convert xyzw → wxyz for body_quat_w format
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    # Joint DOFs: degrees → radians in Newton/MJCF coordinate order. Legacy
    # files already carry this positional order; only their labels were wrong.
    if actual_header == LEGACY_BONES_CSV_HEADER:
        global _LEGACY_HEADER_WARNED
        if not _LEGACY_HEADER_WARNED:
            warnings.warn(
                "Reading legacy GR1T2 SOMA CSV columns positionally because their "
                "head/arm labels are known to be incorrect.",
                RuntimeWarning,
                stacklevel=2,
            )
            _LEGACY_HEADER_WARNED = True
        joint_values = data.iloc[:, 7:].values
    else:
        joint_values = data[BONES_CSV_JOINT_NAMES].values
    joint_pos_mj = np.deg2rad(joint_values).astype(np.float32)
    if joint_pos_mj.shape != (T, NUM_DOF):
        raise ValueError(
            f"GR1T2 DOF shape mismatch for {csv_path}: expected {(T, NUM_DOF)}, "
            f"got {joint_pos_mj.shape}"
        )

    # Create dummy body_pos_w and body_quat_w (only root body populated, rest zeros)
    # The converter only uses body_pos_w[:,0] for root_trans and body_quat_w[:,0] for root_rot
    body_pos_w = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    body_pos_w[:, 0, :] = root_pos
    body_quat_w = np.zeros((T, NUM_BODIES, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0  # identity quaternion wxyz
    body_quat_w[:, 0, :] = root_quat_wxyz

    return {
        "joint_pos": joint_pos_mj,  # (T, 32) canonical MuJoCo order, radians
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,  # wxyz
        "joint_order": "mj",  # already in MuJoCo order, skip IL→MJ reorder
    }


def load_csv_motion(motion_dir: str) -> dict:
    """Load a single motion from a directory of CSV files."""
    joint_pos_f = os.path.join(motion_dir, "joint_pos.csv")
    body_pos_f = os.path.join(motion_dir, "body_pos.csv")
    body_quat_f = os.path.join(motion_dir, "body_quat.csv")

    if not os.path.exists(joint_pos_f):
        return None

    joint_pos = np.loadtxt(joint_pos_f, delimiter=",", skiprows=1, dtype=np.float32)
    body_pos = np.loadtxt(body_pos_f, delimiter=",", skiprows=1, dtype=np.float32)
    body_quat = np.loadtxt(body_quat_f, delimiter=",", skiprows=1, dtype=np.float32)

    # Infer the body count carried by the deploy-format CSV matrices.
    T = joint_pos.shape[0]
    if joint_pos.shape != (T, NUM_DOF):
        raise ValueError(
            f"GR1T2 joint_pos shape mismatch in {motion_dir}: expected {(T, NUM_DOF)}, "
            f"got {joint_pos.shape}"
        )
    body_pos = body_pos.reshape(T, -1, 3)
    body_quat = body_quat.reshape(T, -1, 4)

    return {
        "joint_pos": joint_pos,  # (T, 32) IsaacLab order
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,  # wxyz format
    }


def convert_sequence(seq_data: dict, fps: int, humanoid_fk=None) -> dict:  # noqa: ARG001
    """Convert a single deploy-format sequence to motion_lib format.

    Args:
        seq_data: dict with joint_pos (T, 32), body_pos_w and body_quat_w
        fps: frame rate of the input data
        humanoid_fk: Optional Humanoid_Batch instance (unused, kept for compat)

    Returns:
        motion_lib entry dict with root_trans_offset, pose_aa, dof, root_rot, fps
    """
    joint_pos = np.asarray(seq_data["joint_pos"], dtype=np.float32)
    body_pos_w = np.asarray(seq_data["body_pos_w"], dtype=np.float32)
    body_quat_w = np.asarray(seq_data["body_quat_w"], dtype=np.float32)
    joint_order = seq_data.get("joint_order", "il")  # "il" or "mj"

    T = joint_pos.shape[0]
    if joint_pos.shape != (T, NUM_DOF):
        raise ValueError(f"Expected GR1T2 joint_pos shape {(T, NUM_DOF)}, got {joint_pos.shape}")
    if body_pos_w.ndim != 3 or body_pos_w.shape[0] != T or body_pos_w.shape[2] != 3:
        raise ValueError(f"Invalid body_pos_w shape: {body_pos_w.shape}")
    if body_quat_w.ndim != 3 or body_quat_w.shape[0] != T or body_quat_w.shape[2] != 4:
        raise ValueError(f"Invalid body_quat_w shape: {body_quat_w.shape}")
    if body_pos_w.shape[1] < 1 or body_quat_w.shape[1] < 1:
        raise ValueError("body_pos_w and body_quat_w must contain the root body")
    if not all(np.isfinite(array).all() for array in (joint_pos, body_pos_w, body_quat_w)):
        raise ValueError("GR1T2 input contains NaN or Inf")

    # 1. Root position: body_0 (pelvis) position
    root_trans_offset = body_pos_w[:, 0, :].copy()  # (T, 3)

    # 2. Root quaternion: body_0 quaternion, convert wxyz → xyzw (scipy convention)
    root_quat_wxyz = body_quat_w[:, 0, :]  # (T, 4) [w, x, y, z]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]  # (T, 4) [x, y, z, w]

    # 3. Reorder DOFs to MuJoCo order if needed
    if joint_order == "il":
        # Input is IsaacLab order → reorder to MuJoCo (MJCF actuator order)
        dof_mj = joint_pos[:, MJ_TO_IL]
    elif joint_order == "mj":
        # Input is already in MuJoCo order (e.g., Bones-SEED CSVs)
        dof_mj = joint_pos
    else:
        raise ValueError(f"Unknown joint_order {joint_order!r}; expected 'il' or 'mj'")

    dof = dof_mj
    if dof.shape != (T, NUM_DOF):
        raise ValueError(f"Expected exact GR1T2 DOF shape {(T, NUM_DOF)}, got {dof.shape}")

    # pose_aa[body_idx] = dof_axis * dof_value (axis-angle representation)
    pose_aa = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    for body_index, joint_name in POSE_AA_BODY_TO_JOINT.items():
        dof_index = BODY_JOINTS_MUJOCO.index(joint_name)
        pose_aa[:, body_index, :] = DOF_AXIS[dof_index] * dof[:, dof_index, None]

    # Set root rotation as axis-angle
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_quat_xyzw).as_rotvec()

    return {
        "root_trans_offset": root_trans_offset.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof.astype(np.float32),
        "root_rot": root_quat_xyzw.astype(np.float32),  # xyzw (scipy convention)
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),  # placeholder
        "fps": fps,
    }


def downsample_sequence(entry: dict, fps_source: int, fps_target: int) -> dict:
    """Downsample a motion_lib entry using stride-based frame skipping.

    Matches process_bones_to_motionlib.py: jump = int(fps_source / fps_target).
    Best used when fps_source is an exact multiple of fps_target (e.g. 120→30).
    The resulting PKL is stored at fps_target; fk_batch handles the final
    resampling to target_fps at load time using the canonical interploate_pose formula.
    """
    if fps_source == fps_target:
        return entry
    jump = int(fps_source / fps_target)
    if jump <= 1:
        return entry
    return {
        "root_trans_offset": entry["root_trans_offset"][::jump],
        "pose_aa": entry["pose_aa"][::jump],
        "dof": entry["dof"][::jump],
        "root_rot": entry["root_rot"][::jump],
        "smpl_joints": entry["smpl_joints"][::jump],
        "fps": fps_target,
    }


def init_humanoid_fk():
    """Initialize Humanoid_Batch with the canonical 32 GR1T2 body joints."""
    import omegaconf

    motion_cfg = omegaconf.OmegaConf.create(
        {
            "asset": {
                "assetRoot": "gear_sonic/data/assets/robot_description/mjcf/",
                "assetFileName": "gr1t2_no_fingers.xml",
                "urdfFileName": "",
            },
            "actuated_joint_names": BODY_JOINTS_MUJOCO,
            "extend_config": [],
        }
    )
    from gear_sonic.utils.motion_lib import torch_humanoid_batch

    return torch_humanoid_batch.Humanoid_Batch(motion_cfg)


def process_session_csvs(args_tuple):
    """Process all CSVs in a single session directory. Used by multiprocessing."""
    session_dir, session_name, out_dir, fps, fps_source = args_tuple
    import warnings

    warnings.filterwarnings("ignore")

    csv_files = sorted([f for f in os.listdir(session_dir) if f.endswith(".csv")])

    session_out = os.path.join(out_dir, session_name)
    os.makedirs(session_out, exist_ok=True)

    converted = 0
    failed = 0
    for csv_f in csv_files:
        name = os.path.splitext(csv_f)[0]
        out_path = os.path.join(session_out, name + ".pkl")
        if os.path.exists(out_path):
            converted += 1  # skip existing
            continue
        try:
            seq = load_bones_csv(os.path.join(session_dir, csv_f))
            fps_for_convert = fps_source if fps_source else fps
            entry = convert_sequence(seq, fps_for_convert)
            if fps_source and fps_source != fps:
                entry = downsample_sequence(entry, fps_source, fps)
            joblib.dump({name: entry}, out_path, compress=True)
            converted += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: Failed to convert {os.path.join(session_dir, csv_f)}: {exc}")
            failed += 1
    return session_name, converted, failed, len(csv_files)


def main():
    parser = argparse.ArgumentParser(description="Convert SOMA CSV/PKL to motion_lib format")
    parser.add_argument(
        "--input", required=True, help="CSV dir, parent dir of CSV dirs, or deploy PKL"
    )
    parser.add_argument(
        "--output", required=True, help="Output path (PKL file or directory for individual PKLs)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target output FPS (default: 30, matches process_bones_to_motionlib)",
    )
    parser.add_argument(
        "--fps_source",
        type=int,
        default=None,
        help="Source data FPS. If set and != --fps, data is downsampled. "
        "Bones-SEED CSVs are typically 120fps.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Write individual PKLs per motion (preserves session dir structure)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for --individual mode",
    )
    args = parser.parse_args()

    print(f"GR1T2 {NUM_DOF} DOFs, {NUM_BODIES} MJCF pose bodies")

    # Individual PKL mode: skip scanning, go straight to parallel per-session processing
    if args.individual:
        if not os.path.isdir(args.input):
            print("ERROR: --individual requires a directory input")
            sys.exit(1)

        # Detect: is input a single session dir (contains CSVs) or parent of sessions?
        has_csvs = any(f.endswith(".csv") for f in os.listdir(args.input))
        subdirs = sorted(
            [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
        )
        has_session_subdirs = (
            any(
                any(f.endswith(".csv") for f in os.listdir(os.path.join(args.input, d)))
                for d in subdirs[:3]
            )
            if subdirs
            else False
        )

        session_dirs = []
        if has_session_subdirs:
            for d in subdirs:
                subdir = os.path.join(args.input, d)
                if any(f.endswith(".csv") for f in os.listdir(subdir)):
                    session_dirs.append((subdir, d, args.output, args.fps, args.fps_source))
        elif has_csvs:
            session_name = os.path.basename(args.input.rstrip("/"))
            session_dirs.append((args.input, session_name, args.output, args.fps, args.fps_source))

        print(f"\nBatch converting {len(session_dirs)} sessions with {args.num_workers} workers")
        print(f"Output: {args.output}")
        os.makedirs(args.output, exist_ok=True)

        import multiprocessing

        total_converted = 0
        total_failed = 0
        total_csvs = 0
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            for session_name, converted, failed, n_csvs in pool.imap_unordered(
                process_session_csvs, session_dirs
            ):
                total_converted += converted
                total_failed += failed
                total_csvs += n_csvs
                print(
                    f"  {session_name}: {converted}/{n_csvs} converted"
                    + (f" ({failed} failed)" if failed else "")
                )

        print(
            f"\nDone: {total_converted} motions converted, {total_failed} failed, {total_csvs} total CSVs"
        )
        return

    # Detect input mode (combined PKL output path)
    sequences = {}

    if args.input.endswith(".pkl"):
        # Mode 3: Deploy PKL file
        print(f"Loading deploy PKL: {args.input}")
        data = joblib.load(args.input)
        for name, seq in data.items():
            sequences[name] = seq
        print(f"  Found {len(sequences)} sequences")

    elif os.path.isfile(os.path.join(args.input, "joint_pos.csv")):
        # Mode 1: Single CSV directory
        name = os.path.basename(args.input)
        print(f"Loading single CSV motion: {name}")
        seq = load_csv_motion(args.input)
        if seq is None:
            print("ERROR: joint_pos.csv not found")
            sys.exit(1)
        sequences[name] = seq
        print(f"  {seq['joint_pos'].shape[0]} frames")

    elif os.path.isdir(args.input):
        # Check if directory contains flat CSVs (Bones-SEED format)
        csv_files = sorted([f for f in os.listdir(args.input) if f.endswith(".csv")])
        subdirs = sorted(
            [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
        )

        if csv_files and not any(
            os.path.exists(os.path.join(args.input, d, "joint_pos.csv"))
            for d in subdirs[:5]  # check first 5 subdirs
        ):
            # Mode 4: Directory of flat Bones-SEED CSVs
            print(f"Scanning directory for Bones-SEED CSVs: {args.input}")
            for csv_f in csv_files:
                csv_path = os.path.join(args.input, csv_f)
                name = os.path.splitext(csv_f)[0]
                try:
                    seq = load_bones_csv(csv_path)
                    sequences[name] = seq
                except Exception as e:  # noqa: BLE001
                    print(f"  WARNING: Failed to load {csv_f}: {e}")
            print(f"  Found {len(sequences)} Bones-SEED CSV motions")
        elif subdirs:
            # Check if subdirs contain flat CSVs (batch of session dirs)
            has_session_csvs = False
            for dname in subdirs[:3]:
                subdir = os.path.join(args.input, dname)
                sub_csvs = [f for f in os.listdir(subdir) if f.endswith(".csv")]
                if sub_csvs and not os.path.exists(os.path.join(subdir, "joint_pos.csv")):
                    has_session_csvs = True
                    break

            if has_session_csvs:
                # Mode 5: Parent dir of session dirs containing Bones-SEED CSVs
                print(f"Scanning session directories for Bones-SEED CSVs: {args.input}")
                for dname in sorted(subdirs):
                    subdir = os.path.join(args.input, dname)
                    sub_csvs = sorted([f for f in os.listdir(subdir) if f.endswith(".csv")])
                    for csv_f in sub_csvs:
                        csv_path = os.path.join(subdir, csv_f)
                        name = os.path.splitext(csv_f)[0]
                        try:
                            seq = load_bones_csv(csv_path)
                            sequences[name] = seq
                        except Exception as e:  # noqa: BLE001
                            print(f"  WARNING: Failed to load {dname}/{csv_f}: {e}")
                    if sub_csvs:
                        print(f"  Session {dname}: {len(sub_csvs)} CSVs")
                print(f"  Found {len(sequences)} total Bones-SEED CSV motions")
            else:
                # Mode 2: Parent directory with SOMA-style subdirectories
                print(f"Scanning directory: {args.input}")
                for dname in sorted(subdirs):
                    subdir = os.path.join(args.input, dname)
                    seq = load_csv_motion(subdir)
                    if seq is not None:
                        sequences[dname] = seq
                print(f"  Found {len(sequences)} motion directories with CSVs")
    else:
        print(f"ERROR: {args.input} is not a valid input")
        sys.exit(1)

    if not sequences:
        print("ERROR: No sequences found")
        sys.exit(1)

    # Convert each sequence (combined PKL mode)
    motion_lib_dict = {}
    for name, seq_data in sequences.items():
        T = seq_data["joint_pos"].shape[0]
        print(f"  Converting {name}: {T} frames @ {args.fps} fps")
        fps_for_convert = args.fps_source if args.fps_source else args.fps
        entry = convert_sequence(seq_data, fps_for_convert)
        if args.fps_source and args.fps_source != args.fps:
            entry = downsample_sequence(entry, args.fps_source, args.fps)
        motion_lib_dict[name] = entry

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"\nSaving motion_lib PKL: {args.output}")
    joblib.dump(motion_lib_dict, args.output, compress=True)
    print(f"Done: {len(motion_lib_dict)} sequences saved")


if __name__ == "__main__":
    main()
