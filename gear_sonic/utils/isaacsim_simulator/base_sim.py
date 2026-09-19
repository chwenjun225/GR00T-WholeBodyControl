"""Isaac Sim adapter for Whole-Body-Control in Unitree DDS."""

import time
from pathlib import Path

import yaml # type: ignore
import numpy as np

import omni.usd
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdPhysics

from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager import PhysicsScene
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.experimental.utils.stage import is_stage_loading


from .control import command_torques, joint_groups, map_joints, world_to_body


class BaseSimulator:
    def __init__(self, config, simulation_app):
        self.config = config
        self.app = simulation_app
        self.bridge = None
        self.sim = None
        self.step_count = 0
        self._last_velocity = None
        self._last_dq = None
        self._last_torque = None
        self._active = False
        self._closed = False
        if not omni.usd.get_context().open_stage(str(Path(config.usd_path).expanduser().resolve())):
            raise RuntimeError(f"Cannot open USD scene: {config.usd_path}")
        load_deadline = time.monotonic() + 120
        while is_stage_loading():
            if not self.app.is_running() or time.monotonic() > load_deadline:
                raise TimeoutError("USD dependencies did not finish loading within 120 seconds")
            self.app.update()
        self.stage = omni.usd.get_context().get_stage()
        # Keep runtime edits out of the user's authored scene
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        if UsdGeom.GetStageUpAxis(self.stage) != UsdGeom.Tokens.z:
            raise ValueError("SONIC adapter requires a Z-up scene")
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(self.stage), 1.0):
            raise ValueError("SONIC adapter requires stage units in metres")
        robot_prim = self.stage.GetPrimAtPath(config.robot_path)
        if not robot_prim.IsValid():
            available = [str(p.GetPath()) for p in self.stage.Traverse()
                         if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            raise ValueError(f"Robot prim not found: {config.robot_path}; articulation roots: {available}")
        roots = [p for p in Usd.PrimRange(robot_prim)
                 if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1:
            raise ValueError(f"Expected one articulation root under {config.robot_path}; "
                             f"found {[str(p.GetPath()) for p in roots]}")
        scenes = [p for p in self.stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scenes) != 1:
            raise ValueError(f"Expected one PhysicsScene, found {len(scenes)}")
        def rigid_path(explicit, name):
            candidates = ([self.stage.GetPrimAtPath(explicit)] if explicit else
                          [p for p in Usd.PrimRange(robot_prim) if p.GetName() == name])
            candidates = [p for p in candidates if p.IsValid()
                          and p.HasAPI(UsdPhysics.RigidBodyAPI)
                          and p.GetPath().HasPrefix(robot_prim.GetPath())]
            if len(candidates) != 1:
                raise ValueError(f"Cannot resolve rigid body {name}; supply its explicit path")
            return str(candidates[0].GetPath())
        
        SimulationManager.setup_simulation(dt=config.physics_dt, device="cpu")
        RenderingManager.set_dt(config.physics_dt * config.render_every)

        self.base = RigidPrim(base_path)
        self.torso = RigidPrim(torso_path)

        self.robot = Articulation(str(roots[0].GetPath())) 

        self.sim.reset()
        self.robot.initialize()
        self.base.initialize()
        self.torso.initialize()
        self.groups = joint_groups(config.with_hands)

        self.dof_names = list(self.robot.dof_names)
        self.indices = map_joints(self.dof_names, self.groups)
        self._last_torque = np.zeros(len(self.dof_names), dtype=float)

        controlled = np.unique(np.concatenate(list(self.indices.values()))).astype(np.int32)
        self.robot.switch_dof_control_mode("effort", dof_indices=controlled)

        self.limits = self._effort_limits()
        for group, names in self.groups.items():
            print(f"[{group}] DDS index -> Isaac DOF -> joint")
            for i, (name, index) in enumerate(zip(names, self.indices[group])):
                print(f"  {i:2d} -> {index:2d} -> {name}")
        # print("Joint positions:", self.robot.get_joint_positions())
        print("Joint positions:", self.robot.get_dof_positions())
        print("Base pose (xyz, wxyz):", self.base.get_world_pose())
        print("Torso pose (xyz, wxyz):", self.torso.get_world_pose())

    def _effort_limits(self):
        # This shipped array is interleaved: body[:22], left hand, body[22:], right hand.
        path = Path(__file__).parent / "wbc_configs" / "g1_29dof_sonic_model12.yaml"
        with path.open() as stream:
            raw = np.asarray(yaml.safe_load(stream)["motor_effort_limit_list"], dtype=float)
        if raw.shape != (43,) or not np.isfinite(raw).all() or np.any(raw <= 0):
            raise ValueError("Expected 43 finite positive reference effort limits")
        reference = {"body": np.r_[raw[:22], raw[29:36]],
                     "left_hand": raw[22:29], "right_hand": raw[36:43]}
        usd_limits = np.asarray(self.robot.get_dof_max_efforts(), dtype=float)[0] 
        limits = {group: np.minimum(reference[group], usd_limits[indices])
                  for group, indices in self.indices.items()}
        if any(not np.isfinite(v).all() or np.any(v <= 0) for v in limits.values()):
            raise ValueError("Controlled USD joints must have positive effort limits")
        return limits

    def _observation(self):
        q = np.asarray(self.robot.get_dof_positions(), dtype=float)[0]
        dq = np.asarray(self.robot.get_dof_velocities(), dtype=float)[0]
        
        p, quat = self.base.get_world_pose()
        _, torso_quat = self.torso.get_world_pose()

        base_linear, base_angular = self.base.get_velocities()
        _, torso_angular = self.torso.get_velocities()
        velocity = np.asarray(base_linear, dtype=float)[0]
        angular = np.asarray(base_angular, dtype=float)[0]
        torso_angular = np.asarray(torso_angular, dtype=float)[0]

        arrays = (q, dq, p, quat, torso_quat, velocity, angular, torso_angular)
        if not all(np.isfinite(a).all() for a in arrays):
            raise RuntimeError("Non-finite simulated robot state")
        acceleration = (np.zeros(3) if self._last_velocity is None else
                        (velocity-self._last_velocity)/self.config.physics_dt)
        ddq = (np.zeros_like(dq) if self._last_dq is None else
               (dq-self._last_dq)/self.config.physics_dt)

        
        # Rigid-body linear velocity is world-frame; gyro is body-frame in Unitree
        gravity = self.sim.get_physics_context().get_gravity()
        direction = np.asarray(gravity[0], dtype=float)
        magnitude = float(gravity[1])
        # USD's unauthored gravity magnitude may be -inf (Earth gravity in metre units)
        if not np.isfinite(magnitude):
            magnitude = 9.81
        if np.linalg.norm(direction) < 1e-8:
            direction = np.array([0.0, 0.0, -1.0])
        gravity_world = np.array(self.physics_scene.get_gravity(), dtype=float)



        specific_force = world_to_body(quat, acceleration - gravity_world)
        obs = {
            "time": self.step_count * self.config.physics_dt,
            "floating_base_pose": np.r_[p, quat],
            "floating_base_vel": np.r_[velocity, world_to_body(quat, angular)],
            "floating_base_acc": np.r_[specific_force, np.zeros(3)],
            "odometry_angular_velocity": angular,
            "secondary_imu_quat": torso_quat,
            "secondary_imu_vel": np.r_[np.zeros(3), world_to_body(torso_quat, torso_angular)],
        }
        for group in ("body", "left_hand", "right_hand"):
            idx = self.indices.get(group, np.array([], dtype=np.int32))
            obs[group+"_q"] = q[idx].tolist()
            obs[group+"_dq"] = dq[idx].tolist()
        idx = self.indices["body"]
        obs["body_ddq"] = ddq[idx].tolist()
        # Actuator torque applied at preceding step, not contact/reaction torque
        obs["body_tau_est"] = self._last_torque[idx].tolist()
        self._last_velocity = velocity.copy()
        self._last_dq = dq.copy()
        return obs

    def start(self):
        if self.config.inspect:
            return
        from .simulator_factory import init_channel
        from .unitree_sdk2py_bridge import UnitreeSdk2Bridge

        init_channel(self.config)
        self.bridge = UnitreeSdk2Bridge({
            "ROBOT_TYPE": "g1_29dof", "NUM_MOTORS": 29,
            "NUM_HAND_MOTORS": 7 if self.config.with_hands else 0,
            "USE_SENSOR": False,
        })
        self.bridge.low_state.mode_machine = 5
        print("DDS ready. Physics waits for the first valid body PR command.")
        deadline = time.monotonic()
        waiting_frames = 0
        while self.app.is_running():
            self.bridge.PublishLowState(self._observation())
            snapshot = self.bridge.command_snapshot()
            if snapshot["body"][1] is None:
                if waiting_frames % self.config.render_every == 0:
                    RenderingManager.render()
                waiting_frames += 1
                time.sleep(self.config.physics_dt)
                continue
            now = time.monotonic()
            # A stale body command stops the loop; do not keep driving an old target.
            if now - snapshot["body"][1] > self.config.command_timeout:
                raise TimeoutError("Body DDS command timeout; simulation stopped")

            
            # if not self._active:
            #     controlled = set(np.concatenate(list(self.indices.values())).tolist())
            #     for i in controlled:
            #         self.controller.switch_dof_control_mode(i, "effort")
            #     self._active = True
            #     deadline = now

            q = np.asarray(self.robot.get_dof_positions())[0]
            dq = np.asarray(self.robot.get_dof_velocities())[0]

            for group, indices in self.indices.items():
                message, received_at = snapshot[group]
                if received_at is None or now - received_at > self.config.command_timeout:
                    torque = np.zeros(len(indices))  # Hands may be controlled independently
                else:
                    torque = command_torques(message.motor_cmd, q[indices], dq[indices], self.limits[group])
                    if group == "body":
                        # Body mode 1 enables the servo; hand mode uses packed flags
                        torque = np.where([m.mode == 1 for m in message.motor_cmd[:len(indices)]],torque, 0.0)
                        
                self.robot.set_dof_efforts(torque[np.newaxis, :], dof_indices=indices)

                self._last_torque[indices] = torque
            # Kit Pause/Stop invalidates timing/history assumptions: require a fresh launch
            if not self.sim.is_playing():
                raise RuntimeError("Timeline paused/stopped; restart the adapter")
            self.sim.step(render=False)
            self.step_count += 1
            if self.step_count % self.config.render_every == 0:
                self.sim.render()
            deadline += self.config.physics_dt
            if self.config.realtime:
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.1:
                    print("Warning: simulation is >100 ms behind wall time")
                    deadline = time.monotonic()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.sim is not None:
                self.sim.stop()
        finally:
            if self.bridge is not None:
                self.bridge.close()
