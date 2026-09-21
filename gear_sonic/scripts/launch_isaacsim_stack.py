"""Launch the Isaac Sim + SONIC development stack in one tmux window."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


SESSION_DEFAULT = "cibo_g1"
WINDOW_NAME = "isaac-sonic"
DDS_READY_MARKER = "DDS ready;"


def _run_tmux(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["tmux", *args], check=check, capture_output=True, text=True
    )


def _session_exists(session: str) -> bool:
    return _run_tmux("has-session", "-t", session, check=False).returncode == 0


def _send(session: str, pane: int, command: str) -> None:
    _run_tmux(
        "send-keys", "-t", f"{session}:{WINDOW_NAME}.{pane}", command, "C-m"
    )


def _placeholder(title: str) -> str:
    status = "Status: ĐANG TRONG QUÁ TRÌNH PHÁT TRIỂN"
    return (
        "clear; printf '%s\\n%s\\n' "
        f"{shlex.quote(title)} {shlex.quote(status)}"
    )


def _recorder_status(args: argparse.Namespace) -> str:
    if not args.record_video:
        return _placeholder("Video Recorder / Data Monitor")
    lines = (
        "Recorder worker runs inside the Isaac process",
        f"Output: {args.record_video}",
        f"Capture: {args.video_fps:g} FPS",
        "Encoding: asynchronous process (non-blocking)",
    )
    quoted = " ".join(shlex.quote(line) for line in lines)
    return f"clear; printf '%s\\n%s\\n%s\\n%s\\n' {quoted}"


def _create_session(session: str, working_dir: Path) -> None:
    _run_tmux(
        "new-session", "-d", "-s", session, "-n", WINDOW_NAME, "-c", str(working_dir)
    )
    _run_tmux("set-option", "-t", session, "mouse", "on")
    _run_tmux("set-option", "-t", session, "pane-border-status", "top")

    # Start with left/right, then split both columns to produce a 2x2 grid.
    _run_tmux("split-window", "-t", f"{session}:{WINDOW_NAME}.0", "-h", "-c", str(working_dir))
    _run_tmux("split-window", "-t", f"{session}:{WINDOW_NAME}.0", "-v", "-c", str(working_dir))
    _run_tmux("split-window", "-t", f"{session}:{WINDOW_NAME}.1", "-v", "-c", str(working_dir))
    _run_tmux("select-layout", "-t", f"{session}:{WINDOW_NAME}", "tiled")

    for pane, title in enumerate(("Isaac Sim", "SONIC WBC", "Cosmos3 Policy", "Recorder / Monitor")):
        _run_tmux(
            "select-pane", "-t", f"{session}:{WINDOW_NAME}.{pane}", "-T", title
        )


def _isaac_command(args: argparse.Namespace, repo_root: Path) -> str:
    command = [
        "env",
        "PYTHONUNBUFFERED=1",
        str(args.isaac_python),
        "-m",
        "gear_sonic.scripts.run_isaacsim_loop",
        "--usd-path",
        str(args.usd_path),
        "--robot-path",
        args.robot_path,
        "--base-path",
        args.base_path,
        "--torso-path",
        args.torso_path,
        "--physics-scene-path",
        args.physics_scene_path,
        "--interface",
        args.interface,
        "--domain-id",
        str(args.domain_id),
        "--physics-dt",
        str(args.physics_dt),
        "--render-fps",
        str(args.render_fps),
    ]
    if args.headless:
        command.append("--headless")
    if args.record_video:
        command.extend([
            "--record-video", str(args.record_video),
            "--video-fps", str(args.video_fps),
            "--video-width", str(args.video_width),
            "--video-height", str(args.video_height),
            "--video-camera-path", args.video_camera_path,
        ])
    return f"cd {shlex.quote(str(repo_root))} && {shlex.join(command)}"


def _sonic_wait_command(args: argparse.Namespace, repo_root: Path) -> str:
    isaac_pane = f"{args.session}:{WINDOW_NAME}.0"
    deploy_dir = repo_root / "gear_sonic_deploy"
    deploy_command = shlex.join(["./deploy.sh", *args.sonic_arg, "sim"])
    attempts = max(1, round(args.dds_ready_timeout / 0.5))
    script = (
        "echo 'Waiting for Isaac DDS...'; "
        f"for ((attempt=0; attempt<{attempts}; attempt++)); do "
        f"if tmux capture-pane -p -t {shlex.quote(isaac_pane)} -S -2000 "
        f"| grep -Fq {shlex.quote(DDS_READY_MARKER)}; then "
        "echo 'Isaac DDS ready. Starting SONIC...'; "
        f"cd {shlex.quote(str(deploy_dir))}; exec {deploy_command}; "
        "fi; sleep 0.5; done; "
        f"echo 'ERROR: Isaac DDS was not ready after {args.dds_ready_timeout:g}s.'; "
        "echo 'Inspect the Isaac Sim pane, then restart this pane when ready.'; exec bash"
    )
    return f"bash -lc {shlex.quote(script)}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    cibo_root = script_path.parents[4]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=SESSION_DEFAULT)
    parser.add_argument("--replace", action="store_true",
                        help="Replace an existing tmux session with the same name")
    parser.add_argument("--no-attach", action="store_true",
                        help="Create the session without attaching to it")
    parser.add_argument("--isaac-python", type=Path,
                        default=Path.home() / "isaacsim/python.sh")
    parser.add_argument("--usd-path", type=Path,
                        default=cibo_root / "contents/scenes/cibo.usd")
    parser.add_argument("--robot-path", default="/World/Robots/g1")
    parser.add_argument("--base-path", default="/World/Robots/g1/pelvis")
    parser.add_argument("--torso-path", default="/World/Robots/g1/torso_link")
    parser.add_argument("--physics-scene-path", default="/PhysicsScene")
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--physics-dt", type=float, default=0.002)
    parser.add_argument("--render-fps", type=float, default=25.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record-video", type=Path, default=None,
                        help="Write MP4 asynchronously from the Isaac process")
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-camera-path", default="/OmniverseKit_Persp")
    parser.add_argument("--dds-ready-timeout", type=float, default=180.0)
    parser.add_argument(
        "--sonic-arg", action="append", default=[],
        help="Extra deploy.sh argument; repeat this option for multiple arguments",
    )
    args = parser.parse_args(argv)
    args.repo_root = repo_root
    return args


def _validate(args: argparse.Namespace) -> None:
    errors = []
    if not shutil.which("tmux"):
        errors.append("tmux is not installed")
    if not args.isaac_python.is_file():
        errors.append(f"Isaac Python not found: {args.isaac_python}")
    if not args.usd_path.is_file():
        errors.append(f"USD scene not found: {args.usd_path}")
    if not (args.repo_root / "gear_sonic_deploy/deploy.sh").is_file():
        errors.append("gear_sonic_deploy/deploy.sh not found")
    if (args.physics_dt <= 0 or args.render_fps <= 0 or args.video_fps <= 0
            or args.dds_ready_timeout <= 0):
        errors.append(
            "physics-dt, render-fps, video-fps and dds-ready-timeout must be positive"
        )
    if args.video_width < 1 or args.video_height < 1:
        errors.append("video-width and video-height must be positive")
    if args.record_video:
        args.record_video = args.record_video.expanduser().resolve()
        if args.record_video.suffix.lower() != ".mp4":
            errors.append("record-video must use an .mp4 filename")
    if errors:
        raise SystemExit("Cannot launch:\n  - " + "\n  - ".join(errors))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate(args)

    if _session_exists(args.session):
        if not args.replace:
            print(f"tmux session '{args.session}' already exists.")
            print(f"Attach: tmux attach -t {shlex.quote(args.session)}")
            print("Use --replace to recreate it.")
            return 2
        _run_tmux("kill-session", "-t", args.session)

    _create_session(args.session, args.repo_root)
    _send(args.session, 0, _isaac_command(args, args.repo_root))
    _send(args.session, 1, _sonic_wait_command(args, args.repo_root))
    _send(args.session, 2, _placeholder("Cosmos3-Edge-Policy-G1"))
    _send(args.session, 3, _recorder_status(args))
    _run_tmux("select-pane", "-t", f"{args.session}:{WINDOW_NAME}.0")

    print(f"Created tmux session: {args.session}")
    print("Pane 0: Isaac Sim")
    print("Pane 1: SONIC (waits for DDS ready)")
    print("Pane 2: Cosmos3 placeholder")
    print("Pane 3: Recorder status")
    print(f"Reattach: tmux attach -t {args.session}")

    if not args.no_attach:
        return subprocess.run(["tmux", "attach", "-t", args.session]).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
