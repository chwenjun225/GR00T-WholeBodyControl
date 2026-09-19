"""Standalone Isaac Sim configuration; intentionally independent of MuJoCo/tyro."""
from dataclasses import dataclass
from pathlib import Path
import math


@dataclass
class SimLoopConfig:
    usd_path: str
    robot_path: str
    base_path: str | None = None
    torso_path: str | None = None
    domain_id: int = 0
    interface: str = "lo"
    physics_dt: float = 0.002
    render_every: int = 10
    command_timeout: float = 0.5
    headless: bool = False
    inspect: bool = False
    with_hands: bool = True
    realtime: bool = True

    def __post_init__(self):
        if not Path(self.usd_path).expanduser().is_file():
            raise ValueError(f"USD scene not found: {self.usd_path}")
        for name in ("robot_path", "base_path", "torso_path"):
            value = getattr(self, name)
            if value is not None and (not value.startswith("/") or value == "/"):
                raise ValueError(f"{name} must be an absolute non-root USD prim path")
        for name in ("physics_dt", "command_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.render_every < 1 or not 0 <= self.domain_id <= 232:
            raise ValueError("render_every must be positive; DDS domain must be 0..232")
