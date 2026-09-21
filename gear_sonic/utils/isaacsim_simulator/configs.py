"""Configuration for the Isaac Sim to Unitree DDS simulation loop."""

from dataclasses import dataclass
import math
from pathlib import Path


@dataclass
class SimLoopConfig:
    """Small standalone config for the Isaac Sim process."""

    usd_path: str
    robot_path: str = "/World/Robots/g1"
    base_path: str | None = "/World/Robots/g1/pelvis"
    torso_path: str | None = "/World/Robots/g1/torso_link"
    physics_scene_path: str = "/PhysicsScene"
    domain_id: int = 0
    interface: str | None = "lo"
    physics_dt: float = 0.005
    render_fps: float = 25.0
    # Legacy override. None derives the interval from render_fps.
    render_every: int | None = None
    record_video: str | None = None
    video_fps: float = 20.0
    video_width: int = 640
    video_height: int = 480
    video_camera_path: str = "/OmniverseKit_Persp"
    command_timeout: float = 0.5
    headless: bool = False
    inspect: bool = False
    inspect_only: bool = False
    profile: bool = False
    use_cpp_data_view: bool = True
    # None means infer integrated Dex3 joints from the loaded articulation.
    with_hands: bool | None = None
    realtime: bool = True

    def __post_init__(self) -> None:
        scene = Path(self.usd_path).expanduser()
        if not scene.is_file():
            raise ValueError(f"USD scene not found: {self.usd_path}")
        self.usd_path = str(scene.resolve())
        for name in ("robot_path", "base_path", "torso_path", "physics_scene_path"):
            value = getattr(self, name)
            if value is not None and (not value.startswith("/") or value == "/"):
                raise ValueError(f"{name} must be an absolute non-root USD prim path")
        for name in ("physics_dt", "render_fps", "video_fps", "command_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.render_every is None:
            self.render_every = max(1, round(1.0 / (self.physics_dt * self.render_fps)))
        elif self.render_every < 1:
            raise ValueError("render_every must be positive")
        if self.video_width < 1 or self.video_height < 1:
            raise ValueError("video_width and video_height must be positive")
        if not self.video_camera_path.startswith("/"):
            raise ValueError("video_camera_path must be an absolute USD prim path")
        if self.record_video:
            video_path = Path(self.record_video).expanduser()
            if video_path.suffix.lower() != ".mp4":
                raise ValueError("record_video must use an .mp4 filename")
            self.record_video = str(video_path.resolve())
        if not 0 <= self.domain_id <= 232:
            raise ValueError("DDS domain must be between 0 and 232")
        if self.interface == "":
            self.interface = None
