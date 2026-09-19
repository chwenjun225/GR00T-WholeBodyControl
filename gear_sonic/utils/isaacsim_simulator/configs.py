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
    physics_dt: float = 0.002
    render_every: int = 10
    command_timeout: float = 0.5
    headless: bool = False
    inspect: bool = False
    with_hands: bool = False
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
        for name in ("physics_dt", "command_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.render_every < 1:
            raise ValueError("render_every must be positive")
        if not 0 <= self.domain_id <= 232:
            raise ValueError("DDS domain must be between 0 and 232")
        if self.interface == "":
            self.interface = None
