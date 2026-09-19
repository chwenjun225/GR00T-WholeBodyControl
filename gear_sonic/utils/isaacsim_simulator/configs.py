"""Isaac Sim configuration extending the SONIC simulation-loop config."""

import math 
from dataclasses import dataclass
from pathlib import Path 

from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig as MujocoSimLoopConfig

@dataclass  
class SimLoopConfig(MujocoSimLoopConfig):
    """Simulation loop configuration with IsaacSim options."""
    usd_path: str = ""
    robot_path: str = "World/Robots/g1"

    base_path: str | None = None
    torso_path: str | None = None

    physics_dt: float = 0.002
    render_every: int = 10

    command_timeout: float = 0.5
    headless: bool = False
    inspect: bool = False
    realtime: bool = True

    def __post_init__(self):
        # IMPORTANT: preserve NVIDIA BaseConfig initialization
        super().__post_init__() 

        if not self.usd_path:
            raise ValueError("usd_path is required")
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

        if self.render_every < 1:
            raise ValueError("render_every must be positive")