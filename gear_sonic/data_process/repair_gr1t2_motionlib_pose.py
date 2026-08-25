#!/usr/bin/env python3
"""Repair GR1T2 motion-lib PKLs produced with the legacy SOMA CSV header.

The legacy CSV writer stored Newton's correct positional ``joint_q`` values but
labeled the arm/head columns in a different order. Consequently, the stored
``dof`` array is usable while ``pose_aa`` was scattered onto the wrong bodies.
This tool rebuilds only ``pose_aa`` from the verified MJCF contract and leaves
all other motion fields unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
from scipy.spatial import transform

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (
    BODY_JOINTS_MUJOCO,
    DOF_AXIS,
    DOF_RANGE,
    NUM_BODIES,
    NUM_DOF,
    POSE_AA_BODY_TO_JOINT,
)


def _repair_entry(entry: dict, source: Path) -> dict:
    dof = np.asarray(entry["dof"], dtype=np.float32)
    pose_aa = np.asarray(entry["pose_aa"], dtype=np.float32)
    root_rot = np.asarray(entry["root_rot"], dtype=np.float32)
    num_frames = dof.shape[0]

    if dof.shape != (num_frames, NUM_DOF):
        raise ValueError(f"{source}: expected dof shape (T, {NUM_DOF}), got {dof.shape}")
    if pose_aa.shape != (num_frames, NUM_BODIES, 3):
        raise ValueError(
            f"{source}: expected pose_aa shape (T, {NUM_BODIES}, 3), got {pose_aa.shape}"
        )
    if root_rot.shape != (num_frames, 4):
        raise ValueError(f"{source}: expected root_rot shape (T, 4), got {root_rot.shape}")
    if not np.isfinite(dof).all() or not np.isfinite(root_rot).all():
        raise ValueError(f"{source}: dof/root_rot contains NaN or Inf")

    violation = np.maximum(DOF_RANGE[:, 0] - dof, dof - DOF_RANGE[:, 1])
    max_violation = float(np.maximum(violation, 0.0).max())
    if max_violation > 1e-3:
        raise ValueError(
            f"{source}: dof exceeds the verified MJCF limits by {max_violation:.6f} rad; "
            "this is not the known legacy-header format"
        )

    repaired_pose = np.zeros_like(pose_aa)
    for body_index, joint_name in POSE_AA_BODY_TO_JOINT.items():
        dof_index = BODY_JOINTS_MUJOCO.index(joint_name)
        repaired_pose[:, body_index] = DOF_AXIS[dof_index] * dof[:, dof_index, None]
    repaired_pose[:, 0] = transform.Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)

    repaired = dict(entry)
    repaired["pose_aa"] = repaired_pose
    return repaired


def _repair_file(task: tuple[Path, Path, bool]) -> tuple[str, float]:
    source, destination, dry_run = task
    loaded = joblib.load(source)
    if "pose_aa" in loaded:
        repaired = _repair_entry(loaded, source)
    else:
        repaired = {
            key: _repair_entry(entry, source)
            for key, entry in loaded.items()
        }

    old_entries = [loaded] if "pose_aa" in loaded else loaded.values()
    new_entries = [repaired] if "pose_aa" in repaired else repaired.values()
    max_change = max(
        float(np.max(np.abs(old["pose_aa"] - new["pose_aa"])))
        for old, new in zip(old_entries, new_entries, strict=True)
    )
    if dry_run:
        return str(source), max_change

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(repaired, temporary_path, compress=True)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(source), max_change


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input PKL file or directory")
    parser.add_argument("--output", type=Path, help="Output file or directory")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Explicitly allow atomic replacement of input files",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing files")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.in_place and args.output is not None:
        parser.error("--in-place and --output are mutually exclusive")
    if not args.in_place and not args.dry_run and args.output is None:
        parser.error("provide --output, or explicitly select --in-place")

    sources = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.pkl"))
    if not sources:
        raise ValueError(f"No PKL files found under {input_path}")

    if args.in_place or args.dry_run:
        destinations = sources
    else:
        output_path = args.output.resolve()
        if input_path.is_file():
            destinations = [output_path]
        else:
            destinations = [output_path / source.relative_to(input_path) for source in sources]

    tasks = [(source, destination, args.dry_run) for source, destination in zip(
        sources, destinations, strict=True
    )]
    changed = 0
    max_pose_change = 0.0
    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        for index, (_, pose_change) in enumerate(executor.map(_repair_file, tasks, chunksize=16), 1):
            changed += pose_change > 1e-6
            max_pose_change = max(max_pose_change, pose_change)
            if index % 1000 == 0 or index == len(tasks):
                print(f"Processed {index}/{len(tasks)} files")

    mode = "Validated" if args.dry_run else "Repaired"
    print(
        f"{mode} {len(tasks)} files; changed={changed}; "
        f"max_pose_change={max_pose_change:.6f} rad"
    )


if __name__ == "__main__":
    main()
