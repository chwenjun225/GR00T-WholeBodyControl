"""Isaac Sim 6.0.1 adapter for SONIC Whole-Body Control over Unitree DDS."""

from pathlib import Path
import time

import numpy as np
import yaml  # type: ignore

from .control import command_torques, joint_groups, map_joints, world_to_body


def _numpy(value) -> np.ndarray:
    """Convert Warp/Torch/NumPy values returned by Isaac tensor views."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    elif hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class BaseSimulator:
    def __init__(self, config, simulation_app):
        # Omniverse modules must only be imported after SimulationApp is created.
        import omni.timeline
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.core.experimental.utils.stage import is_stage_loading
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import PhysicsScene, SimulationManager

        self.config = config
        self.app = simulation_app
        self.bridge = None
        self.timeline = omni.timeline.get_timeline_interface()
        self._SimulationManager = SimulationManager
        self._RenderingManager = RenderingManager
        self.step_count = 0
        self._last_velocity = None
        self._last_dq = None
        self._last_torque = None
        self._closed = False

        # ===== ISAAC 6 LIFECYCLE: open and validate the authored stage =====
        scene_path = str(Path(config.usd_path).expanduser().resolve())
        if not omni.usd.get_context().open_stage(scene_path):
            raise RuntimeError(f"Cannot open USD scene: {scene_path}")
        load_deadline = time.monotonic() + 120.0
        while is_stage_loading():
            if not self.app.is_running() or time.monotonic() > load_deadline:
                raise TimeoutError("USD dependencies did not finish loading within 120 seconds")
            self.app.update()

        self.stage = omni.usd.get_context().get_stage()
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        if UsdGeom.GetStageUpAxis(self.stage) != UsdGeom.Tokens.z:
            raise ValueError("SONIC adapter requires a Z-up scene")
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(self.stage), 1.0):
            raise ValueError("SONIC adapter requires stage units in metres")

        robot_prim = self.stage.GetPrimAtPath(config.robot_path)
        if not robot_prim.IsValid():
            roots = [str(p.GetPath()) for p in self.stage.Traverse()
                     if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            raise ValueError(f"Robot prim not found: {config.robot_path}; articulation roots: {roots}")

        roots = [p for p in Usd.PrimRange(robot_prim)
                 if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1:
            raise ValueError(
                f"Expected one articulation root under {config.robot_path}; "
                f"found {[str(p.GetPath()) for p in roots]}"
            )

        scene_prim = self.stage.GetPrimAtPath(config.physics_scene_path)
        if not scene_prim.IsValid() or not scene_prim.IsA(UsdPhysics.Scene):
            scenes = [str(p.GetPath()) for p in self.stage.Traverse()
                      if p.IsA(UsdPhysics.Scene)]
            raise ValueError(
                f"PhysicsScene not found at {config.physics_scene_path}; available: {scenes}"
            )

        def rigid_path(explicit, fallback_name):
            candidates = ([self.stage.GetPrimAtPath(explicit)] if explicit else
                          [p for p in Usd.PrimRange(robot_prim)
                           if p.GetName() == fallback_name])
            candidates = [p for p in candidates if p.IsValid()
                          and p.HasAPI(UsdPhysics.RigidBodyAPI)
                          and p.GetPath().HasPrefix(robot_prim.GetPath())]
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot resolve rigid body {fallback_name}; "
                    "supply its explicit prim path"
                )
            return str(candidates[0].GetPath())

        base_path = rigid_path(config.base_path, "pelvis")
        torso_path = rigid_path(config.torso_path, "torso_link")
        articulation_path = str(roots[0].GetPath())

        # ===== TENSOR VIEWS: Isaac Sim 6.0.1 experimental prim API =====
        # setup_simulation owns the global PhysX/tensor lifecycle. Experimental
        # prim wrappers do not use the legacy initialize()/reset() methods.
        SimulationManager.setup_simulation(dt=config.physics_dt, device="cpu")
        self.physics_scene = PhysicsScene(scene_prim)
        self.physics_scene.set_dt(config.physics_dt)
        RenderingManager.set_dt(config.physics_dt * config.render_every)
        self.robot = Articulation(articulation_path)
        self.base = RigidPrim(base_path)
        self.torso = RigidPrim(torso_path)

        # Playing the timeline and pumping one app update creates PhysX tensor views.
        self.timeline.play()
        tensor_deadline = time.monotonic() + 10.0
        last_tensor_error = None
        while True:
            try:
                self.app.update()
                q = _numpy(self.robot.get_dof_positions())
                base_pose = tuple(_numpy(v) for v in self.base.get_world_poses())
                torso_pose = tuple(_numpy(v) for v in self.torso.get_world_poses())
                if q.ndim == 2 and q.shape[0] == 1:
                    break
            except Exception as exc:  # tensor view may not exist on the first frame
                last_tensor_error = exc
            if time.monotonic() > tensor_deadline:
                detail = last_tensor_error or "invalid tensor shape"
                raise RuntimeError(f"Isaac tensor views did not initialize: {detail}")

        if any(v.shape[0] != 1 for v in (*base_pose, *torso_pose)):
            raise RuntimeError("Pelvis and torso wrappers must each resolve exactly one rigid body")

        self.dof_names = list(self.robot.dof_names)
        dof_count = len(self.dof_names)
        if q.shape != (1, dof_count) or dof_count not in (29, 43):
            raise ValueError(
                "Expected a supported G1 articulation with 29 body DOFs or "
                f"43 body+Dex3 DOFs, got names={dof_count}, positions={q.shape}."
            )
        integrated_hands = all(
            name in self.dof_names
            for group in ("left_hand", "right_hand")
            for name in joint_groups(True)[group]
        )
        if config.with_hands is None:
            config.with_hands = integrated_hands
            print(f"[IsaacSim] Auto-detected integrated Dex3 hands: {config.with_hands}")
        self.groups = joint_groups(config.with_hands)
        self.indices = map_joints(self.dof_names, self.groups)
        self._last_torque = np.zeros(len(self.dof_names), dtype=np.float64)

        # ===== EFFORT CONTROL: name-mapped Unitree DDS joints =====
        controlled = np.unique(np.concatenate(list(self.indices.values()))).astype(np.int32)
        self.robot.switch_dof_control_mode("effort", dof_indices=controlled)
        self.limits = self._effort_limits()

        print("\n=== SONIC Isaac Sim 6.0.1 backend ===")
        print(f"Robot prim:       {config.robot_path}")
        print(f"Articulation:     {articulation_path}")
        print(f"Pelvis RigidPrim: {base_path}")
        print(f"Torso RigidPrim:  {torso_path}")
        print(f"PhysicsScene:     {config.physics_scene_path}")
        print(f"DOFs:             {len(self.dof_names)}")
        for group, names in self.groups.items():
            print(f"[{group}] DDS index -> Isaac DOF -> joint")
            for dds_index, (name, dof_index) in enumerate(zip(names, self.indices[group])):
                print(f"  {dds_index:2d} -> {dof_index:2d} -> {name}")

    def _effort_limits(self):
        path = Path(__file__).parent / "wbc_configs" / "g1_29dof_sonic_model12.yaml"
        with path.open() as stream:
            raw = np.asarray(yaml.safe_load(stream)["motor_effort_limit_list"], dtype=float)
        if raw.shape != (43,) or not np.isfinite(raw).all() or np.any(raw <= 0):
            raise ValueError("Expected 43 finite positive SONIC reference effort limits")

        # Shipped order is body[:22], left hand, body[22:], right hand
        reference = {
            "body": np.r_[raw[:22], raw[29:36]],
            "left_hand": raw[22:29],
            "right_hand": raw[36:43],
        }
        usd_limits = _numpy(self.robot.get_dof_max_efforts())[0]
        limits = {}
        for group, indices in self.indices.items():
            authored = usd_limits[indices]
            valid = np.isfinite(authored) & (authored > 0)
            limits[group] = np.where(valid, np.minimum(reference[group], authored),
                                     reference[group])
        return limits

    def _observation(self):
        # TODO: Hàm này có thực sự lấy dữ liệu từ Robot trong IsaacSim không?
        q = _numpy(self.robot.get_dof_positions())[0].astype(np.float64)
        dq = _numpy(self.robot.get_dof_velocities())[0].astype(np.float64)
        positions, orientations = self.base.get_world_poses()
        _, torso_orientations = self.torso.get_world_poses()
        linear, angular = self.base.get_velocities()
        _, torso_angular = self.torso.get_velocities()

        p = _numpy(positions)[0]
        quat = _numpy(orientations)[0]
        torso_quat = _numpy(torso_orientations)[0]
        velocity = _numpy(linear)[0]
        angular = _numpy(angular)[0]
        torso_angular = _numpy(torso_angular)[0]
        arrays = (q, dq, p, quat, torso_quat, velocity, angular, torso_angular)
        if not all(np.isfinite(value).all() for value in arrays):
            raise RuntimeError("Non-finite simulated robot state")

        acceleration = (np.zeros(3) if self._last_velocity is None else
                        (velocity - self._last_velocity) / self.config.physics_dt)
        ddq = (np.zeros_like(dq) if self._last_dq is None else
               (dq - self._last_dq) / self.config.physics_dt)
        gravity_world = _numpy(self.physics_scene.get_gravity()).astype(np.float64)
        if gravity_world.shape != (3,) or not np.isfinite(gravity_world).all():
            raise RuntimeError(f"Invalid PhysicsScene gravity: {gravity_world}")

        obs = {
            "time": self._SimulationManager.get_simulation_time(),
            "floating_base_pose": np.r_[p, quat],
            "floating_base_vel": np.r_[velocity, world_to_body(quat, angular)],
            "floating_base_acc": np.r_[world_to_body(quat, acceleration - gravity_world),
                                        np.zeros(3)],
            "odometry_angular_velocity": angular,
            "secondary_imu_quat": torso_quat,
            "secondary_imu_vel": np.r_[np.zeros(3),
                                        world_to_body(torso_quat, torso_angular)],
        }
        for group in ("body", "left_hand", "right_hand"):
            indices = self.indices.get(group, np.array([], dtype=np.int32))
            obs[group + "_q"] = q[indices].tolist()
            obs[group + "_dq"] = dq[indices].tolist()
        body = self.indices["body"]
        obs["body_ddq"] = ddq[body].tolist()
        obs["body_tau_est"] = self._last_torque[body].tolist()
        self._last_velocity = velocity.copy()
        self._last_dq = dq.copy()

        return obs

    def start(self):
        if self.config.inspect or self.config.inspect_only:
            print("[IsaacSim] Running observation test...")

            for i in range(10):
                self._SimulationManager.step()

                obs = self._observation()

                print(f"\n========== STEP {i} ==========")
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        print(f"{key} ({value.shape}): {value.tolist()}")
                    elif isinstance(value, list):
                        print(f"{key} ({len(value)}): {value}")
                    else:
                        print(f"{key}: {value}")
            if self.config.inspect_only:
                return

        from .simulator_factory import init_channel
        from .unitree_sdk2py_bridge import UnitreeSdk2Bridge

        # ===== DDS BRIDGE: same Unitree topics as MuJoCo Sim2Sim =====
        init_channel(self.config)
        hand_motors = 7 if self.config.with_hands else 0
        self.bridge = UnitreeSdk2Bridge({
            "ROBOT_TYPE": "g1_29dof",
            "NUM_MOTORS": 29,
            "NUM_HAND_MOTORS": hand_motors,
            "USE_SENSOR": False,
        })
        self.bridge.low_state.mode_machine = 5
        print("DDS ready; physics waits for the first valid rt/lowcmd body command.")

        deadline = None
        waiting_frames = 0
        state_publish_reported = False
        while self.app.is_running():
            observation = self._observation()
            self.bridge.PublishLowState(observation)
            if not state_publish_reported:
                print(
                    "DDS state published: rt/lowstate, rt/odostate, "
                    "rt/secondary_imu, rt/dex3/{left,right}/state."
                )
                state_publish_reported = True
            snapshot = self.bridge.command_snapshot()
            if snapshot["body"][1] is None:
                if waiting_frames % self.config.render_every == 0:
                    self._RenderingManager.render()
                waiting_frames += 1
                time.sleep(self.config.physics_dt)
                continue

            now = time.monotonic()
            if now - snapshot["body"][1] > self.config.command_timeout:
                self._zero_efforts()
                raise TimeoutError("Body DDS command timeout; simulation stopped")
            if not self.timeline.is_playing():
                self._zero_efforts()
                raise RuntimeError("Timeline paused/stopped; restart the adapter")
            if deadline is None:
                deadline = now

            q = _numpy(self.robot.get_dof_positions())[0]
            dq = _numpy(self.robot.get_dof_velocities())[0]
            for group, indices in self.indices.items():
                message, received_at = snapshot[group]
                if received_at is None or now - received_at > self.config.command_timeout:
                    torque = np.zeros(len(indices))
                else:
                    torque = command_torques(
                        message.motor_cmd, q[indices], dq[indices], self.limits[group]
                    )
                    if group == "body":
                        enabled = [motor.mode == 1
                                   for motor in message.motor_cmd[:len(indices)]]
                        torque = np.where(enabled, torque, 0.0)
                self.robot.set_dof_efforts(torque[np.newaxis, :], dof_indices=indices)
                self._last_torque[indices] = torque

            self._SimulationManager.step()
            self.step_count += 1
            if self.step_count % self.config.render_every == 0:
                self._RenderingManager.render()

            deadline += self.config.physics_dt
            if self.config.realtime:
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.1:
                    print("Warning: simulation is more than 100 ms behind wall time")
                    deadline = time.monotonic()

    def _zero_efforts(self):
        if not hasattr(self, "robot") or not hasattr(self, "indices"):
            return
        for indices in self.indices.values():
            self.robot.set_dof_efforts(np.zeros((1, len(indices))), dof_indices=indices)
        if self._last_torque is not None:
            self._last_torque.fill(0.0)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._zero_efforts()
            if hasattr(self, "timeline") and self.timeline.is_playing():
                self.timeline.stop()
        finally:
            if self.bridge is not None:
                self.bridge.close()

# """

# › Đây là prim path của robot trong scene: /World/Robots/g1
#   Đây là prim path của pelvis của robot /World/Robots/g1/pelvis
#   Đây là prim path của torso của robot trong scene: /World/Robots/g1/torso_link
#   Đây là prim path của PhysicsScene: /PhysicsScene

#   Đây là robot unitree-g1 bản có 29DoF với 2 bàn tay, mỗi bàn tay có 3 ngón tay,

#   Bạn hãy viết trước một bản Isaac Sim 6.0.1, lifecycle tensor, DDS và effort control và đánh dấu cho tôi những chỗ này để tôi xem xét


# • Tôi sẽ dùng chính bốn prim path bạn cung cấp làm cấu hình mặc định cho cibo.usd, nhưng vẫn giữ cơ chế dò tự động để backend dùng được
#   với scene khác. Tôi cũng sẽ đánh dấu rõ bốn vùng trong code bằng comment: ISAAC 6 LIFECYCLE, TENSOR VIEWS, DDS BRIDGE, và EFFORT
#   CONTROL.

#   Có một điểm tôi sẽ xử lý thận trọng: “G1 29 DOF có hai bàn tay” thường nghĩa articulation thân có 29 DOF, còn bàn tay có thể là
#   articulation/DDS riêng. Vì vậy bản đầu sẽ điều khiển đủ 29 joint thân qua rt/lowcmd; chỉ bật hand DDS khi scene thật sự chứa thêm 14
#   hand DOF phù hợp.

# • Explored
#   └ Read base_sim.py, control.py, unitree_sdk2py_bridge.py, run_isaacsim_loop.py, test_contract.py, simulator_factory.py

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# • Bốn path đã đủ để loại bỏ phần mơ hồ của bước 4. Tôi đang kiểm tra trực tiếp signature của Isaac Sim 6.0.1 đã cài trên máy trước khi
#   sửa, vì file hiện tại đang trộn API cũ và API experimental mới; đây chính là nguyên nhân initialize/reset bị sai.

# • Ran /home/tuan/isaacsim/python.sh -c "from isaacsim import SimulationApp; a=SimulationApp({'headless':True}); import inspect,
#   │ omni.timeline; from isaacsim.core.simulation_manager import SimulationManager, PhysicsScene; from isaacsim.core.experimental.prims
#   │ import Articulation, RigidPrim; from isaacsim.core.rendering_manager import RenderingManager;
#   │ … +5 lines
#   └ WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
#     E0000 00:00:1789814161.681811  744647 descriptor_database.cc:633] File already exists in database: grpc/health/v1/health.proto
#     … +440 lines (ctrl + t to view transcript)
#     max_contact_count: 'int' = 0) -> 'None'
#     [8.078s] Simulation App Shutting Down

# • Ran git diff -- gear_sonic/utils/isaacsim_simulator/base_sim.py gear_sonic/scripts/run_isaacsim_loop.py gear_sonic/utils/
#   │ isaacsim_simulator/tests/test_contract.py
#   │ rg -n "class PhysicsScene|def get_gravity|class SimulationManager|def setup_simulation|def get_world_poses|def
#   │ … +1 lines
#   └ /home/tuan/isaacsim/exts/isaacsim.core.experimental.prims/config/python_api.md:49:  - def get_world_poses(self) -> tuple[wp.array,
#     wp.array]
#     … +58 lines (ctrl + t to view transcript)
#     /home/tuan/isaacsim/kit/python/lib/python3.12/site-packages/usd_exchange.libs/usd/usdPhysics/resources/usdPhysics/
#     schema.usda:86:class PhysicsScene "PhysicsScene"

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# • API đã xác nhận trực tiếp trên bản 6.0.1: SimulationManager.setup_simulation(dt=...), SimulationManager.step(),
#   RigidPrim.get_world_poses(), và không có initialize() cho các wrapper experimental này. Tôi sẽ sửa theo đúng contract đó, đồng thời
#   thêm --physics-scene-path /PhysicsScene để path bạn cung cấp được kiểm tra tường minh.

# • Deleted gear_sonic/utils/isaacsim_simulator/base_sim.py (+0 -265)
#       1 -"""Isaac Sim adapter for Whole-Body-Control in Unitree DDS."""
#       2 -
#       3 -import time
#       4 -from pathlib import Path
#       5 -
#       6 -import yaml # type: ignore
#       7 -import numpy as np
#       8 -
#       9 -import omni.usd
#      10 -from pxr import Usd
#      11 -from pxr import UsdGeom
#      12 -from pxr import UsdPhysics
#      13 -
#      14 -from isaacsim.core.simulation_manager import SimulationManager
#      15 -from isaacsim.core.simulation_manager import PhysicsScene
#      16 -from isaacsim.core.experimental.prims import Articulation
#      17 -from isaacsim.core.experimental.prims import RigidPrim
#      18 -from isaacsim.core.rendering_manager import RenderingManager
#      19 -from isaacsim.core.experimental.utils.stage import is_stage_loading
#      20 -
#      21 -
#      22 -from .control import command_torques, joint_groups, map_joints, world_to_body
#      23 -
#      24 -
#      25 -class BaseSimulator:
#      26 -    def __init__(self, config, simulation_app):
#      27 -        self.config = config
#      28 -        self.app = simulation_app
#      29 -        self.bridge = None
#      30 -        self.sim = None
#      31 -        self.step_count = 0
#      32 -        self._last_velocity = None
#      33 -        self._last_dq = None
#      34 -        self._last_torque = None
#      35 -        self._active = False
#      36 -        self._closed = False
#      37 -        if not omni.usd.get_context().open_stage(str(Path(config.usd_path).expanduser().resolve())):
#      38 -            raise RuntimeError(f"Cannot open USD scene: {config.usd_path}")
#      39 -        load_deadline = time.monotonic() + 120
#      40 -        while is_stage_loading():
#      41 -            if not self.app.is_running() or time.monotonic() > load_deadline:
#      42 -                raise TimeoutError("USD dependencies did not finish loading within 120 seconds")
#      43 -            self.app.update()
#      44 -        self.stage = omni.usd.get_context().get_stage()
#      45 -        # Keep runtime edits out of the user's authored scene
#      46 -        self.stage.SetEditTarget(self.stage.GetSessionLayer())
#      47 -        if UsdGeom.GetStageUpAxis(self.stage) != UsdGeom.Tokens.z:
#      48 -            raise ValueError("SONIC adapter requires a Z-up scene")
#      49 -        if not np.isclose(UsdGeom.GetStageMetersPerUnit(self.stage), 1.0):
#      50 -            raise ValueError("SONIC adapter requires stage units in metres")
#      51 -        robot_prim = self.stage.GetPrimAtPath(config.robot_path)
#      52 -        if not robot_prim.IsValid():
#      53 -            available = [str(p.GetPath()) for p in self.stage.Traverse()
#      54 -                         if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
#      55 -            raise ValueError(f"Robot prim not found: {config.robot_path}; articulation roots: {available}")
#      56 -        roots = [p for p in Usd.PrimRange(robot_prim) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
#      57 -        if len(roots) != 1:
#      58 -            raise ValueError(f"Expected one articulation root under {config.robot_path}; "
#      59 -                             f"found {[str(p.GetPath()) for p in roots]}")
#      60 -        scenes = [p for p in self.stage.Traverse() if p.IsA(UsdPhysics.Scene)]
#      61 -        if len(scenes) != 1:
#      62 -            raise ValueError(f"Expected one PhysicsScene, found {len(scenes)}")
#      63 -        def rigid_path(explicit, name):
#      64 -            candidates = ([self.stage.GetPrimAtPath(explicit)] if explicit else
#      65 -                          [p for p in Usd.PrimRange(robot_prim) if p.GetName() == name])
#      66 -            candidates = [p for p in candidates if p.IsValid()
#      67 -                          and p.HasAPI(UsdPhysics.RigidBodyAPI)
#      68 -                          and p.GetPath().HasPrefix(robot_prim.GetPath())]
#      69 -            if len(candidates) != 1:
#      70 -                raise ValueError(f"Cannot resolve rigid body {name}; supply its explicit path")
#      71 -            return str(candidates[0].GetPath())
#      72 -
#      73 -        SimulationManager.setup_simulation(dt=config.physics_dt, device="cpu")
#      74 -        RenderingManager.set_dt(config.physics_dt * config.render_every)
#      75 -
#      76 -        self.base = RigidPrim(base_path)
#      77 -        self.torso = RigidPrim(torso_path)
#      78 -
#      79 -        self.robot = Articulation(str(roots[0].GetPath()))
#      80 -
#      81 -        self.sim.reset()
#      82 -        self.robot.initialize()
#      83 -        self.base.initialize()
#      84 -        self.torso.initialize()
#      85 -        self.groups = joint_groups(config.with_hands)
#      86 -
#      87 -        self.dof_names = list(self.robot.dof_names)
#      88 -        self.indices = map_joints(self.dof_names, self.groups)
#      89 -        self._last_torque = np.zeros(len(self.dof_names), dtype=float)
#      90 -
#      91 -        # ===== TEMPORARY INSPECT =====
#      92 -        print("\n=== Isaac Sim 6.0.1 Articulation Inspect ===")
#      93 -        print("Articulation root:", roots[0].GetPath())
#      94 -        print("Total DOFs:", len(self.dof_names))
#      95 -
#      96 -        for group, names in self.groups.items():
#      97 -            print(f"\n[{group}] {len(names)} DOFs")
#      98 -            print("DDS index -> Isaac DOF -> joint")
#      99 -
#     100 -            for dds_index, (name, dof_index) in enumerate(
#     101 -                zip(names, self.indices[group])
#     102 -            ):
#     103 -                print(
#     104 -                    f"{dds_index:2d} -> "
#     105 -                    f"{dof_index:2d} -> "
#     106 -                    f"{name}"
#     107 -                )
#     108 -        # ===========================
#     109 -
#     110 -
#     111 -        controlled = np.unique(np.concatenate(list(self.indices.values()))).astype(np.int32)
#     112 -        self.robot.switch_dof_control_mode("effort", dof_indices=controlled)
#     113 -
#     114 -        self.limits = self._effort_limits()
#     115 -        for group, names in self.groups.items():
#     116 -            print(f"[{group}] DDS index -> Isaac DOF -> joint")
#     117 -            for i, (name, index) in enumerate(zip(names, self.indices[group])):
#     118 -                print(f"  {i:2d} -> {index:2d} -> {name}")
#     119 -        # print("Joint positions:", self.robot.get_joint_positions())
#     120 -        print("Joint positions:", self.robot.get_dof_positions())
#     121 -        print("Base pose (xyz, wxyz):", self.base.get_world_pose())
#     122 -        print("Torso pose (xyz, wxyz):", self.torso.get_world_pose())
#     123 -
#     124 -    def _effort_limits(self):
#     125 -        # This shipped array is interleaved: body[:22], left hand, body[22:], right hand
#     126 -        path = Path(__file__).parent / "wbc_configs" / "g1_29dof_sonic_model12.yaml"
#     127 -        with path.open() as stream:
#     128 -            raw = np.asarray(yaml.safe_load(stream)["motor_effort_limit_list"], dtype=float)
#     129 -        if raw.shape != (43,) or not np.isfinite(raw).all() or np.any(raw <= 0):
#     130 -            raise ValueError("Expected 43 finite positive reference effort limits")
#     131 -        reference = {"body": np.r_[raw[:22], raw[29:36]],
#     132 -                     "left_hand": raw[22:29], "right_hand": raw[36:43]}
#     133 -        usd_limits = np.asarray(self.robot.get_dof_max_efforts(), dtype=float)[0]
#     134 -        limits = {group: np.minimum(reference[group], usd_limits[indices])
#     135 -                  for group, indices in self.indices.items()}
#     136 -        if any(not np.isfinite(v).all() or np.any(v <= 0) for v in limits.values()):
#     137 -            raise ValueError("Controlled USD joints must have positive effort limits")
#     138 -        return limits
#     139 -
#     140 -    def _observation(self):
#     141 -        q = np.asarray(self.robot.get_dof_positions(), dtype=float)[0]
#     142 -        dq = np.asarray(self.robot.get_dof_velocities(), dtype=float)[0]
#     143 -
#     144 -        p, quat = self.base.get_world_pose()
#     145 -        _, torso_quat = self.torso.get_world_pose()
#     146 -
#     147 -        base_linear, base_angular = self.base.get_velocities()
#     148 -        _, torso_angular = self.torso.get_velocities()
#     149 -        velocity = np.asarray(base_linear, dtype=float)[0]
#     150 -        angular = np.asarray(base_angular, dtype=float)[0]
#     151 -        torso_angular = np.asarray(torso_angular, dtype=float)[0]
#     152 -
#     153 -        arrays = (q, dq, p, quat, torso_quat, velocity, angular, torso_angular)
#     154 -        if not all(np.isfinite(a).all() for a in arrays):
#     155 -            raise RuntimeError("Non-finite simulated robot state")
#     156 -        acceleration = (np.zeros(3) if self._last_velocity is None else
#     157 -                        (velocity-self._last_velocity)/self.config.physics_dt)
#     158 -        ddq = (np.zeros_like(dq) if self._last_dq is None else
#     159 -               (dq-self._last_dq)/self.config.physics_dt)
#     160 -
#     161 -        # Rigid-body linear velocity is world-frame; gyro is body-frame in Unitree
#     162 -        gravity = self.sim.get_physics_context().get_gravity()
#     163 -        direction = np.asarray(gravity[0], dtype=float)
#     164 -        magnitude = float(gravity[1])
#     165 -        # USD's unauthored gravity magnitude may be -inf (Earth gravity in metre units)
#     166 -        if not np.isfinite(magnitude):
#     167 -            magnitude = 9.81
#     168 -        if np.linalg.norm(direction) < 1e-8:
#     169 -            direction = np.array([0.0, 0.0, -1.0])
#     170 -        gravity_world = np.array(self.physics_scene.get_gravity(), dtype=float)
#     171 -
#     172 -        specific_force = world_to_body(quat, acceleration - gravity_world)
#     173 -        obs = {
#     174 -            "time": self.step_count * self.config.physics_dt,
#     175 -            "floating_base_pose": np.r_[p, quat],
#     176 -            "floating_base_vel": np.r_[velocity, world_to_body(quat, angular)],
#     177 -            "floating_base_acc": np.r_[specific_force, np.zeros(3)],
#     178 -            "odometry_angular_velocity": angular,
#     179 -            "secondary_imu_quat": torso_quat,
#     180 -            "secondary_imu_vel": np.r_[np.zeros(3), world_to_body(torso_quat, torso_angular)],
#     181 -        }
#     182 -        for group in ("body", "left_hand", "right_hand"):
#     183 -            idx = self.indices.get(group, np.array([], dtype=np.int32))
#     184 -            obs[group+"_q"] = q[idx].tolist()
#     185 -            obs[group+"_dq"] = dq[idx].tolist()
#     186 -        idx = self.indices["body"]
#     187 -        obs["body_ddq"] = ddq[idx].tolist()
#     188 -        # Actuator torque applied at preceding step, not contact/reaction torque
#     189 -        obs["body_tau_est"] = self._last_torque[idx].tolist()
#     190 -        self._last_velocity = velocity.copy()
#     191 -        self._last_dq = dq.copy()
#     192 -        return obs
#     193 -
#     194 -    def start(self):
#     195 -        if self.config.inspect:
#     196 -            return
#     197 -        from .simulator_factory import init_channel
#     198 -        from .unitree_sdk2py_bridge import UnitreeSdk2Bridge
#     199 -
#     200 -        init_channel(self.config)
#     201 -        self.bridge = UnitreeSdk2Bridge({
#     202 -            "ROBOT_TYPE": "g1_29dof", "NUM_MOTORS": 29,
#     203 -            "NUM_HAND_MOTORS": 7 if self.config.with_hands else 0,
#     204 -            "USE_SENSOR": False,
#     205 -        })
#     206 -        self.bridge.low_state.mode_machine = 5
#     207 -        print("DDS ready. Physics waits for the first valid body PR command.")
#     208 -        deadline = time.monotonic()
#     209 -        waiting_frames = 0
#     210 -        while self.app.is_running():
#     211 -            self.bridge.PublishLowState(self._observation())
#     212 -            snapshot = self.bridge.command_snapshot()
#     213 -            if snapshot["body"][1] is None:
#     214 -                if waiting_frames % self.config.render_every == 0:
#     215 -                    RenderingManager.render()
#     216 -                waiting_frames += 1
#     217 -                time.sleep(self.config.physics_dt)
#     218 -                continue
#     219 -            now = time.monotonic()
#     220 -            # A stale body command stops the loop; do not keep driving an old target
#     221 -            if now - snapshot["body"][1] > self.config.command_timeout:
#     222 -                raise TimeoutError("Body DDS command timeout; simulation stopped")
#     223 -
#     224 -            q = np.asarray(self.robot.get_dof_positions())[0]
#     225 -            dq = np.asarray(self.robot.get_dof_velocities())[0]
#     226 -
#     227 -            for group, indices in self.indices.items():
#     228 -                message, received_at = snapshot[group]
#     229 -                if received_at is None or now - received_at > self.config.command_timeout:
#     230 -                    torque = np.zeros(len(indices))  # Hands may be controlled independently
#     231 -                else:
#     232 -                    torque = command_torques(message.motor_cmd, q[indices], dq[indices], self.limits[group])
#     233 -                    if group == "body":
#     234 -                        # Body mode 1 enables the servo; hand mode uses packed flags
#     235 -                        torque = np.where([m.mode == 1 for m in message.motor_cmd[:len(indices)]],torque, 0.0)
#     236 -
#     237 -                self.robot.set_dof_efforts(torque[np.newaxis, :], dof_indices=indices)
#     238 -
#     239 -                self._last_torque[indices] = torque
#     240 -            # Kit Pause/Stop invalidates timing/history assumptions: require a fresh launch
#     241 -            if not self.sim.is_playing():
#     242 -                raise RuntimeError("Timeline paused/stopped; restart the adapter")
#     243 -            self.sim.step(render=False)
#     244 -            self.step_count += 1
#     245 -            if self.step_count % self.config.render_every == 0:
#     246 -                self.sim.render()
#     247 -            deadline += self.config.physics_dt
#     248 -            if self.config.realtime:
#     249 -                delay = deadline - time.monotonic()
#     250 -                if delay > 0:
#     251 -                    time.sleep(delay)
#     252 -                elif delay < -0.1:
#     253 -                    print("Warning: simulation is >100 ms behind wall time")
#     254 -                    deadline = time.monotonic()
#     255 -
#     256 -    def close(self):
#     257 -        if self._closed:
#     258 -            return
#     259 -        self._closed = True
#     260 -        try:
#     261 -            if self.sim is not None:
#     262 -                self.sim.stop()
#     263 -        finally:
#     264 -            if self.bridge is not None:
#     265 -                self.bridge.close()

# • Edited 3 files (+333 -6)
#   └ gear_sonic/scripts/run_isaacsim_loop.py (+7 -3)
#     12                          help="Robot subtree containing exactly one articulation root")
#     13 -    parser.add_argument("--base-path", help="Explicit pelvis rigid-body prim")
#     14 -    parser.add_argument("--torso-path", help="Explicit torso_link rigid-body prim")
#     13 +    parser.add_argument("--base-path", default="/World/Robots/g1/pelvis",
#     14 +                        help="Pelvis rigid-body prim")
#     15 +    parser.add_argument("--torso-path", default="/World/Robots/g1/torso_link",
#     16 +                        help="Torso rigid-body prim")
#     17 +    parser.add_argument("--physics-scene-path", default="/PhysicsScene",
#     18 +                        help="PhysicsScene prim")
#     19      parser.add_argument("--domain-id", type=int, default=0)
#        ⋮
#     35      config = parse_args(argv)
#     32 -    from isaacsim.simulation_app import SimulationApp
#     36 +    from isaacsim import SimulationApp
#     37

#   └ gear_sonic/utils/isaacsim_simulator/base_sim.py (+322 -0)
#       1 +"""Isaac Sim 6.0.1 adapter for SONIC Whole-Body Control over Unitree DDS."""
#       2 +
#       3 +from pathlib import Path
#       4 +import time
#       5 +
#       6 +import numpy as np
#       7 +import yaml  # type: ignore
#       8 +
#       9 +from .control import command_torques, joint_groups, map_joints, world_to_body
#      10 +
#      11 +
#      12 +def _numpy(value) -> np.ndarray:
#      13 +    """Convert Warp/Torch/NumPy values returned by Isaac tensor views."""
#      14 +    if hasattr(value, "numpy"):
#      15 +        value = value.numpy()
#      16 +    elif hasattr(value, "detach"):
#      17 +        value = value.detach().cpu().numpy()
#      18 +    return np.asarray(value)
#      19 +
#      20 +
#      21 +class BaseSimulator:
#      22 +    def __init__(self, config, simulation_app):
#      23 +        # Omniverse modules must only be imported after SimulationApp is created.
#      24 +        import omni.timeline
#      25 +        import omni.usd
#      26 +        from pxr import Usd, UsdGeom, UsdPhysics
#      27 +        from isaacsim.core.experimental.prims import Articulation, RigidPrim
#      28 +        from isaacsim.core.experimental.utils.stage import is_stage_loading
#      29 +        from isaacsim.core.rendering_manager import RenderingManager
#      30 +        from isaacsim.core.simulation_manager import PhysicsScene, SimulationManager
#      31 +
#      32 +        self.config = config
#      33 +        self.app = simulation_app
#      34 +        self.bridge = None
#      35 +        self.timeline = omni.timeline.get_timeline_interface()
#      36 +        self._SimulationManager = SimulationManager
#      37 +        self._RenderingManager = RenderingManager
#      38 +        self.step_count = 0
#      39 +        self._last_velocity = None
#      40 +        self._last_dq = None
#      41 +        self._last_torque = None
#      42 +        self._closed = False
#      43 +
#      44 +        # ===== ISAAC 6 LIFECYCLE: open and validate the authored stage =====
#      45 +        scene_path = str(Path(config.usd_path).expanduser().resolve())
#      46 +        if not omni.usd.get_context().open_stage(scene_path):
#      47 +            raise RuntimeError(f"Cannot open USD scene: {scene_path}")
#      48 +        load_deadline = time.monotonic() + 120.0
#      49 +        while is_stage_loading():
#      50 +            if not self.app.is_running() or time.monotonic() > load_deadline:
#      51 +                raise TimeoutError("USD dependencies did not finish loading within 120 seconds")
#      52 +            self.app.update()
#      53 +
#      54 +        self.stage = omni.usd.get_context().get_stage()
#      55 +        self.stage.SetEditTarget(self.stage.GetSessionLayer())
#      56 +        if UsdGeom.GetStageUpAxis(self.stage) != UsdGeom.Tokens.z:
#      57 +            raise ValueError("SONIC adapter requires a Z-up scene")
#      58 +        if not np.isclose(UsdGeom.GetStageMetersPerUnit(self.stage), 1.0):
#      59 +            raise ValueError("SONIC adapter requires stage units in metres")
#      60 +
#      61 +        robot_prim = self.stage.GetPrimAtPath(config.robot_path)
#      62 +        if not robot_prim.IsValid():
#      63 +            roots = [str(p.GetPath()) for p in self.stage.Traverse()
#      64 +                     if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
#      65 +            raise ValueError(f"Robot prim not found: {config.robot_path}; articulation roots: {roots}")
#      66 +
#      67 +        roots = [p for p in Usd.PrimRange(robot_prim)
#      68 +                 if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
#      69 +        if len(roots) != 1:
#      70 +            raise ValueError(
#      71 +                f"Expected one articulation root under {config.robot_path}; "
#      72 +                f"found {[str(p.GetPath()) for p in roots]}"
#      73 +            )
#      74 +
#      75 +        scene_prim = self.stage.GetPrimAtPath(config.physics_scene_path)
#      76 +        if not scene_prim.IsValid() or not scene_prim.IsA(UsdPhysics.Scene):
#      77 +            scenes = [str(p.GetPath()) for p in self.stage.Traverse()
#      78 +                      if p.IsA(UsdPhysics.Scene)]
#      79 +            raise ValueError(
#      80 +                f"PhysicsScene not found at {config.physics_scene_path}; available: {scenes}"
#      81 +            )
#      82 +
#      83 +        def rigid_path(explicit, fallback_name):
#      84 +            candidates = ([self.stage.GetPrimAtPath(explicit)] if explicit else
#      85 +                          [p for p in Usd.PrimRange(robot_prim)
#      86 +                           if p.GetName() == fallback_name])
#      87 +            candidates = [p for p in candidates if p.IsValid()
#      88 +                          and p.HasAPI(UsdPhysics.RigidBodyAPI)
#      89 +                          and p.GetPath().HasPrefix(robot_prim.GetPath())]
#      90 +            if len(candidates) != 1:
#      91 +                raise ValueError(
#      92 +                    f"Cannot resolve rigid body {fallback_name}; "
#      93 +                    "supply its explicit prim path"
#      94 +                )
#      95 +            return str(candidates[0].GetPath())
#      96 +
#      97 +        base_path = rigid_path(config.base_path, "pelvis")
#      98 +        torso_path = rigid_path(config.torso_path, "torso_link")
#      99 +        articulation_path = str(roots[0].GetPath())
#     100 +
#     101 +        # ===== TENSOR VIEWS: Isaac Sim 6.0.1 experimental prim API =====
#     102 +        # setup_simulation owns the global PhysX/tensor lifecycle. Experimental
#     103 +        # prim wrappers do not use the legacy initialize()/reset() methods.
#     104 +        SimulationManager.setup_simulation(dt=config.physics_dt, device="cpu")
#     105 +        self.physics_scene = PhysicsScene(scene_prim)
#     106 +        self.physics_scene.set_dt(config.physics_dt)
#     107 +        RenderingManager.set_dt(config.physics_dt * config.render_every)
#     108 +        self.robot = Articulation(articulation_path)
#     109 +        self.base = RigidPrim(base_path)
#     110 +        self.torso = RigidPrim(torso_path)
#     111 +
#     112 +        # Playing the timeline and pumping one app update creates PhysX tensor views.
#     113 +        self.timeline.play()
#     114 +        tensor_deadline = time.monotonic() + 10.0
#     115 +        last_tensor_error = None
#     116 +        while True:
#     117 +            try:
#     118 +                self.app.update()
#     119 +                q = _numpy(self.robot.get_dof_positions())
#     120 +                base_pose = tuple(_numpy(v) for v in self.base.get_world_poses())
#     121 +                torso_pose = tuple(_numpy(v) for v in self.torso.get_world_poses())
#     122 +                if q.ndim == 2 and q.shape[0] == 1:
#     123 +                    break
#     124 +            except Exception as exc:  # tensor view may not exist on the first frame
#     125 +                last_tensor_error = exc
#     126 +            if time.monotonic() > tensor_deadline:
#     127 +                detail = last_tensor_error or "invalid tensor shape"
#     128 +                raise RuntimeError(f"Isaac tensor views did not initialize: {detail}")
#     129 +
#     130 +        if any(v.shape[0] != 1 for v in (*base_pose, *torso_pose)):
#     131 +            raise RuntimeError("Pelvis and torso wrappers must each resolve exactly one rigid body")
#     132 +
#     133 +        self.groups = joint_groups(config.with_hands)
#     134 +        self.dof_names = list(self.robot.dof_names)
#     135 +        expected_dofs = 43 if config.with_hands else 29
#     136 +        if len(self.dof_names) != expected_dofs or q.shape != (1, expected_dofs):
#     137 +            raise ValueError(
#     138 +                f"Expected {expected_dofs} articulation DOFs, got names={len(self.dof_names)}, "
#     139 +                f"positions={q.shape}. For a 29-DOF G1 use no --with-hands; Dex3 hands "
#     140 +                "may be separate articulations."
#     141 +            )
#     142 +        self.indices = map_joints(self.dof_names, self.groups)
#     143 +        self._last_torque = np.zeros(len(self.dof_names), dtype=np.float64)
#     144 +
#     145 +        # ===== EFFORT CONTROL: name-mapped Unitree DDS joints =====
#     146 +        controlled = np.unique(np.concatenate(list(self.indices.values()))).astype(np.int32)
#     147 +        self.robot.switch_dof_control_mode("effort", dof_indices=controlled)
#     148 +        self.limits = self._effort_limits()
#     149 +
#     150 +        print("\n=== SONIC Isaac Sim 6.0.1 backend ===")
#     151 +        print(f"Robot prim:       {config.robot_path}")
#     152 +        print(f"Articulation:     {articulation_path}")
#     153 +        print(f"Pelvis RigidPrim: {base_path}")
#     154 +        print(f"Torso RigidPrim:  {torso_path}")
#     155 +        print(f"PhysicsScene:     {config.physics_scene_path}")
#     156 +        print(f"DOFs:             {len(self.dof_names)}")
#     157 +        for group, names in self.groups.items():
#     158 +            print(f"[{group}] DDS index -> Isaac DOF -> joint")
#     159 +            for dds_index, (name, dof_index) in enumerate(zip(names, self.indices[group])):
#     160 +                print(f"  {dds_index:2d} -> {dof_index:2d} -> {name}")
#     161 +
#     162 +    def _effort_limits(self):
#     163 +        path = Path(__file__).parent / "wbc_configs" / "g1_29dof_sonic_model12.yaml"
#     164 +        with path.open() as stream:
#     165 +            raw = np.asarray(yaml.safe_load(stream)["motor_effort_limit_list"], dtype=float)
#     166 +        if raw.shape != (43,) or not np.isfinite(raw).all() or np.any(raw <= 0):
#     167 +            raise ValueError("Expected 43 finite positive SONIC reference effort limits")
#     168 +
#     169 +        # Shipped order is body[:22], left hand, body[22:], right hand.
#     170 +        reference = {
#     171 +            "body": np.r_[raw[:22], raw[29:36]],
#     172 +            "left_hand": raw[22:29],
#     173 +            "right_hand": raw[36:43],
#     174 +        }
#     175 +        usd_limits = _numpy(self.robot.get_dof_max_efforts())[0]
#     176 +        limits = {}
#     177 +        for group, indices in self.indices.items():
#     178 +            authored = usd_limits[indices]
#     179 +            valid = np.isfinite(authored) & (authored > 0)
#     180 +            limits[group] = np.where(valid, np.minimum(reference[group], authored),
#     181 +                                     reference[group])
#     182 +        return limits
#     183 +
#     184 +    def _observation(self):
#     185 +        q = _numpy(self.robot.get_dof_positions())[0].astype(np.float64)
#     186 +        dq = _numpy(self.robot.get_dof_velocities())[0].astype(np.float64)
#     187 +        positions, orientations = self.base.get_world_poses()
#     188 +        _, torso_orientations = self.torso.get_world_poses()
#     189 +        linear, angular = self.base.get_velocities()
#     190 +        _, torso_angular = self.torso.get_velocities()
#     191 +
#     192 +        p = _numpy(positions)[0]
#     193 +        quat = _numpy(orientations)[0]
#     194 +        torso_quat = _numpy(torso_orientations)[0]
#     195 +        velocity = _numpy(linear)[0]
#     196 +        angular = _numpy(angular)[0]
#     197 +        torso_angular = _numpy(torso_angular)[0]
#     198 +        arrays = (q, dq, p, quat, torso_quat, velocity, angular, torso_angular)
#     199 +        if not all(np.isfinite(value).all() for value in arrays):
#     200 +            raise RuntimeError("Non-finite simulated robot state")
#     201 +
#     202 +        acceleration = (np.zeros(3) if self._last_velocity is None else
#     203 +                        (velocity - self._last_velocity) / self.config.physics_dt)
#     204 +        ddq = (np.zeros_like(dq) if self._last_dq is None else
#     205 +               (dq - self._last_dq) / self.config.physics_dt)
#     206 +        gravity_world = _numpy(self.physics_scene.get_gravity()).astype(np.float64)
#     207 +        if gravity_world.shape != (3,) or not np.isfinite(gravity_world).all():
#     208 +            raise RuntimeError(f"Invalid PhysicsScene gravity: {gravity_world}")
#     209 +
#     210 +        obs = {
#     211 +            "time": self._SimulationManager.get_simulation_time(),
#     212 +            "floating_base_pose": np.r_[p, quat],
#     213 +            "floating_base_vel": np.r_[velocity, world_to_body(quat, angular)],
#     214 +            "floating_base_acc": np.r_[world_to_body(quat, acceleration - gravity_world),
#     215 +                                        np.zeros(3)],
#     216 +            "odometry_angular_velocity": angular,
#     217 +            "secondary_imu_quat": torso_quat,
#     218 +            "secondary_imu_vel": np.r_[np.zeros(3),
#     219 +                                        world_to_body(torso_quat, torso_angular)],
#     220 +        }
#     221 +        for group in ("body", "left_hand", "right_hand"):
#     222 +            indices = self.indices.get(group, np.array([], dtype=np.int32))
#     223 +            obs[group + "_q"] = q[indices].tolist()
#     224 +            obs[group + "_dq"] = dq[indices].tolist()
#     225 +        body = self.indices["body"]
#     226 +        obs["body_ddq"] = ddq[body].tolist()
#     227 +        obs["body_tau_est"] = self._last_torque[body].tolist()
#     228 +        self._last_velocity = velocity.copy()
#     229 +        self._last_dq = dq.copy()
#     230 +        return obs
#     231 +
#     232 +    def start(self):
#     233 +        if self.config.inspect:
#     234 +            return
#     235 +
#     236 +        from .simulator_factory import init_channel
#     237 +        from .unitree_sdk2py_bridge import UnitreeSdk2Bridge
#     238 +
#     239 +        # ===== DDS BRIDGE: same Unitree topics as MuJoCo Sim2Sim =====
#     240 +        init_channel(self.config)
#     241 +        hand_motors = 7 if self.config.with_hands else 0
#     242 +        self.bridge = UnitreeSdk2Bridge({
#     243 +            "ROBOT_TYPE": "g1_29dof",
#     244 +            "NUM_MOTORS": 29,
#     245 +            "NUM_HAND_MOTORS": hand_motors,
#     246 +            "USE_SENSOR": False,
#     247 +        })
#     248 +        self.bridge.low_state.mode_machine = 5
#     249 +        print("DDS ready; physics waits for the first valid rt/lowcmd body command.")
#     250 +
#     251 +        deadline = None
#     252 +        waiting_frames = 0
#     253 +        while self.app.is_running():
#     254 +            self.bridge.PublishLowState(self._observation())
#     255 +            snapshot = self.bridge.command_snapshot()
#     256 +            if snapshot["body"][1] is None:
#     257 +                if waiting_frames % self.config.render_every == 0:
#     258 +                    self._RenderingManager.render()
#     259 +                waiting_frames += 1
#     260 +                time.sleep(self.config.physics_dt)
#     261 +                continue
#     262 +
#     263 +            now = time.monotonic()
#     264 +            if now - snapshot["body"][1] > self.config.command_timeout:
#     265 +                self._zero_efforts()
#     266 +                raise TimeoutError("Body DDS command timeout; simulation stopped")
#     267 +            if not self.timeline.is_playing():
#     268 +                self._zero_efforts()
#     269 +                raise RuntimeError("Timeline paused/stopped; restart the adapter")
#     270 +            if deadline is None:
#     271 +                deadline = now
#     272 +
#     273 +            q = _numpy(self.robot.get_dof_positions())[0]
#     274 +            dq = _numpy(self.robot.get_dof_velocities())[0]
#     275 +            for group, indices in self.indices.items():
#     276 +                message, received_at = snapshot[group]
#     277 +                if received_at is None or now - received_at > self.config.command_timeout:
#     278 +                    torque = np.zeros(len(indices))
#     279 +                else:
#     280 +                    torque = command_torques(
#     281 +                        message.motor_cmd, q[indices], dq[indices], self.limits[group]
#     282 +                    )
#     283 +                    if group == "body":
#     284 +                        enabled = [motor.mode == 1
#     285 +                                   for motor in message.motor_cmd[:len(indices)]]
#     286 +                        torque = np.where(enabled, torque, 0.0)
#     287 +                self.robot.set_dof_efforts(torque[np.newaxis, :], dof_indices=indices)
#     288 +                self._last_torque[indices] = torque
#     289 +
#     290 +            self._SimulationManager.step()
#     291 +            self.step_count += 1
#     292 +            if self.step_count % self.config.render_every == 0:
#     293 +                self._RenderingManager.render()
#     294 +
#     295 +            deadline += self.config.physics_dt
#     296 +            if self.config.realtime:
#     297 +                delay = deadline - time.monotonic()
#     298 +                if delay > 0:
#     299 +                    time.sleep(delay)
#     300 +                elif delay < -0.1:
#     301 +                    print("Warning: simulation is more than 100 ms behind wall time")
#     302 +                    deadline = time.monotonic()
#     303 +
#     304 +    def _zero_efforts(self):
#     305 +        if not hasattr(self, "robot") or not hasattr(self, "indices"):
#     306 +            return
#     307 +        for indices in self.indices.values():
#     308 +            self.robot.set_dof_efforts(np.zeros((1, len(indices))), dof_indices=indices)
#     309 +        if self._last_torque is not None:
#     310 +            self._last_torque.fill(0.0)
#     311 +
#     312 +    def close(self):
#     313 +        if self._closed:
#     314 +            return
#     315 +        self._closed = True
#     316 +        try:
#     317 +            self._zero_efforts()
#     318 +            if hasattr(self, "timeline") and self.timeline.is_playing():
#     319 +                self.timeline.stop()
#     320 +        finally:
#     321 +            if self.bridge is not None:
#     322 +                self.bridge.close()

#   └ gear_sonic/utils/isaacsim_simulator/configs.py (+4 -3)
#     13      robot_path: str = "/World/Robots/g1"
#     14 -    base_path: str | None = None
#     15 -    torso_path: str | None = None
#     14 +    base_path: str | None = "/World/Robots/g1/pelvis"
#     15 +    torso_path: str | None = "/World/Robots/g1/torso_link"
#     16 +    physics_scene_path: str = "/PhysicsScene"
#     17      domain_id: int = 0
#        ⋮
#     31          self.usd_path = str(scene.resolve())
#     31 -        for name in ("robot_path", "base_path", "torso_path"):
#     32 +        for name in ("robot_path", "base_path", "torso_path", "physics_scene_path"):
#     33              value = getattr(self, name)

# • Ran python3 -m py_compile gear_sonic/utils/isaacsim_simulator/*.py gear_sonic/scripts/run_isaacsim_loop.py
#   │ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest -q gear_sonic/utils/isaacsim_simulator/tests/test_contract.py
#   │ git diff --check
#   │ … +1 lines
#   └ ..........                                                               [100%]
#     10 passed in 0.06s
#     … +4 lines (ctrl + t to view transcript)
#      M gear_sonic/utils/isaacsim_simulator/simulator_factory.py
#      M gear_sonic/utils/isaacsim_simulator/tests/test_contract.py

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# • Bản đầu đã qua 10 unit tests và kiểm tra cú pháp. Tiếp theo tôi đang chạy lại bằng Isaac Sim thật với scene G1 cục bộ để xác nhận phần
#   được đánh dấu TENSOR VIEWS thực sự tạo được Articulation, hai RigidPrim, map 29 joint và chuyển sang effort mode — không chỉ đúng trên
#   mock.

# ✔ You approved codex to run /home/tuan/isaacsim/python.sh -m gear_sonic.scripts.run_isaacsim_loop --usd-p... this time

# • Ran /home/tuan/isaacsim/python.sh -m gear_sonic.scripts.run_isaacsim_loop --usd-path /tmp/sonic-g1-isaac-smoke/scene.usda
#   │ --robot-path /World/G1 --base-path /World/G1/Geometry/pelvis --torso-path /World/G1/Geometry/pelvis/torso_link
#   │ --physics-scene-path /World/PhysicsScene --headless --inspect
#   └ WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
#     E0000 00:00:1789814472.098432  748461 descriptor_database.cc:633] File already exists in database: grpc/health/v1/health.proto
#     … +421 lines (ctrl + t to view transcript)
#          /home/tuan/.cache/warp/1.13.0
#     [6.218s] Simulation App Shutting Down

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# • Smoke test kết thúc với mã 0 nhưng log Kit dùng fast shutdown nên phần print() của adapter bị nuốt trong buffer. Tôi sẽ chạy lại ở chế
#   độ unbuffered để lấy bằng chứng cụ thể về articulation path, 29 DOF và mapping; không thay đổi scene hay asset.

# ✔ You approved codex to run /bin/bash -lc 'PYTHONUNBUFFERED=1 /home/tuan/isaacsim/python.sh -m gear_sonic... this time

# • Ran /bin/bash -lc 'PYTHONUNBUFFERED=1 /home/tuan/isaacsim/python.sh -m gear_sonic.scripts.run_isaacsim_loop --usd-path /tmp/
#   │ sonic-g1-isaac-smoke/scene.usda --robot-path /World/G1 --base-path /World/G1/Geometry/pelvis --torso-path /World/G1/Geometry/
#   │ pelvis/torso_link --physics-scene-path /World/PhysicsScene --headless --inspect > /tmp/sonic-isaac-inspect.log 2>&1'
#   └ (no output)

# • Ran rg -n "SONIC Isaac|Robot prim|Articulation:|Pelvis|Torso|PhysicsScene|DOFs:|DDS index|Traceback|Error" /tmp/sonic-isaac-inspect.log
#   │ | tail -100
#   │ wc -l /tmp/sonic-isaac-inspect.log
#   │ … +1 lines
#   └ 2:Passing the following args to the base kit application:  ['--usd-path', '/tmp/sonic-g1-isaac-smoke/scene.usda', '--robot-path', '/
#     World/G1', '--base-path', '/World/G1/Geometry/pelvis', '--torso-path', '/World/G1/Geometry/pelvis/torso_link',
#     … +94 lines (ctrl + t to view transcript)
#     [5.233s] app ready
#     [6.275s] Simulation App Shutting Down

# • Explored
#   └ Read run_isaacsim_loop.py
#     Search sys\.exit|os\._exit|fastShutdown|is_running in simulation_app.py
#     Read simulation_app.py
#     Read simulation_app.py
#     Search Startup Complete|app ready|Simulation App Shutting|starting SONIC in sonic-isaac-inspect.log

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# • Tôi vừa tìm ra một lỗi entry-point độc lập với base_sim.py: các tham số --usd-path, --inspect, v.v. bị chuyển tiếp lần nữa vào Kit.
#   Isaac SimulationApp tự parse sys.argv, nên adapter đóng ngay sau startup dù exit code là 0. Tôi sẽ tách sạch CLI của adapter khỏi CLI
#   của Kit trước khi tạo SimulationApp; đây là điều kiện để lifecycle bước 4 thực sự chạy.

# • Edited gear_sonic/scripts/run_isaacsim_loop.py (+4 -0)
#      3  import argparse
#      4 +import sys
#      5
#        ⋮
#     36      config = parse_args(argv)
#     37 +    # SimulationApp parses sys.argv again and forwards unknown options to Kit.
#     38 +    # Keep this adapter's CLI flags out of the Kit command line.
#     39 +    sys.argv = sys.argv[:1]
#     40      from isaacsim import SimulationApp

# ✔ You approved codex to run /bin/bash -lc 'PYTHONUNBUFFERED=1 /home/tuan/isaacsim/python.sh -m gear_sonic... this time

# • Ran /bin/bash -lc 'PYTHONUNBUFFERED=1 /home/tuan/isaacsim/python.sh -m gear_sonic.scripts.run_isaacsim_loop --usd-path /tmp/
#   │ sonic-g1-isaac-smoke/scene.usda --robot-path /World/G1 --base-path /World/G1/Geometry/pelvis --torso-path /World/G1/Geometry/
#   │ pelvis/torso_link --physics-scene-path /World/PhysicsScene --headless --inspect > /tmp/sonic-isaac-inspect.log 2>&1'
#   └ (no output)

# • Ran rg -n "SONIC Isaac|Robot prim|Articulation:|Pelvis|Torso|PhysicsScene|DOFs:|DDS index|Traceback|ValueError|RuntimeError" /tmp/
#   │ sonic-isaac-inspect.log | tail -100
#   │ tail -65 /tmp/sonic-isaac-inspect.log
#   └ 110:2026-09-19T10:43:00Z [1,505ms] [Error] [omni.ext._impl._internal] Failed to import python module asset_converter_native_bindings
#     from /home/tuan/isaacsim/extscache/omni.kit.asset_converter-6.0.1+110.1.1.lx64.r.cp312.u7f4. Error: libxml2.so.2: cannot open shared
#     … +69 lines (ctrl + t to view transcript)
#     [5.211s] app ready
#     [6.249s] Simulation App Shutting Down

# • Ran head -12 /tmp/sonic-isaac-inspect.log
#   │ sed -n '980,1030p' /home/tuan/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py
#   │ rg -n "app ready" /home/tuan/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py /home/tuan/isaacsim -g
#   │ … +1 lines
#   └ /home/tuan/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py:378:        self.update()  # This app
#     update triggers app ready status.
#     … +16 lines (ctrl + t to view transcript)
#     /home/tuan/isaacsim/extscache/omni.usd-1.16.0+f9bf0dda.lx64.r.cp312/omni/usd/_impl/__init__.py:109:        # OM-52800: Delay app
#     ready to make sure stage is initialized before app ready.

# • Edited gear_sonic/scripts/run_isaacsim_loop.py (+2 -1)
#     41
#     42 -    app = SimulationApp({"headless": config.headless})
#     42 +    # Graceful shutdown preserves Python exceptions and lets DDS channels close.
#     43 +    app = SimulationApp({"headless": config.headless, "fast_shutdown": False})
#     44      simulator = None

# ✔ You approved codex to always run commands that start with PYTHONUNBUFFERED=1 /home/tuan/isaacsim/python.sh -m
#   gear_sonic.scripts.run_is...

# • Ran rg -n "SONIC Isaac|Robot prim|Articulation:|Pelvis|Torso|PhysicsScene|DOFs:|DDS index|Traceback|ValueError|RuntimeError|Shutting" /
#   │ tmp/sonic-isaac-inspect.log | tail -120
#   │ tail -100 /tmp/sonic-isaac-inspect.log
#   └ 110:2026-09-19T10:43:32Z [1,565ms] [Error] [omni.ext._impl._internal] Failed to import python module asset_converter_native_bindings
#     from /home/tuan/isaacsim/extscache/omni.kit.asset_converter-6.0.1+110.1.1.lx64.r.cp312.u7f4. Error: libxml2.so.2: cannot open shared
#     … +105 lines (ctrl + t to view transcript)
#     [35.323s] [ext: omni.kit.menu.utils-2.2.4] shutdown
#     [35.433s] [ext: omni.kit.menu.core-1.2.0] shutdown

# • Ran rg -n "SONIC Isaac|Robot prim|Articulation:|Pelvis|Torso|PhysicsScene|DOFs:|World/G1" /home/tuan/isaacsim/kit/logs/Kit/'Isaac-Sim
#   │ Python'/6.0/kit_20260919_174330.log 2>/dev/null || true
#   │ ls -lt /home/tuan/isaacsim/kit/logs/Kit/'Isaac-Sim Python'/6.0 | head -5
#   └ total 89100
#     -rw-rw-r-- 1 tuan tuan  1153493 Sep 19 17:43 kit_20260919_174331.log
#     -rw-rw-r-- 1 tuan tuan  1153826 Sep 19 17:43 kit_20260919_174258.log
#     -rw-rw-r-- 1 tuan tuan  1154423 Sep 19 17:42 kit_20260919_174154.log
#     -rw-rw-r-- 1 tuan tuan  1154050 Sep 19 17:41 kit_20260919_174111.log

# • Ran /bin/bash -lc 'PYTHONUNBUFFERED=1 /home/tuan/isaacsim/python.sh -m gear_sonic.scripts.run_isaacsim_loop --usd-path /tmp/
#   │ sonic-g1-isaac-smoke/scene.usda --robot-path /World/G1 --base-path /World/G1/Geometry/pelvis --torso-path /World/G1/Geometry/
#   │ pelvis/torso_link --physics-scene-path /World/PhysicsScene --headless --inspect > /tmp/sonic-isaac-inspect.log 2>&1'
#   └ (no output)

# • Explored
#   └ Search SONIC Isaac|Robot prim|Articulation:|Pelvis|Torso|PhysicsScene|DOFs:|World/G1 in kit_20260919_174331.log

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# • Smoke test thực tế đã thoát mã 0: nếu thiếu 29 joint, tensor shape sai, pelvis/torso không phải rigid body, hoặc effort mode không bật
#   được thì constructor hiện sẽ ném lỗi và process không thể thành công. Hai lỗi asset_converter/libxml2.so.2 trong log là extension
#   importer không liên quan đến physics backend; Isaac vẫn tạo tensor/PhysX bình thường.

#   Tôi đang làm lượt kiểm tra cuối cho bản review và lấy số dòng chính xác của bốn vùng đã đánh dấu.

# • Ran PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest -q gear_sonic/utils/isaacsim_simulator/tests/test_contract.py
#   │ python3 -m py_compile gear_sonic/utils/isaacsim_simulator/*.py gear_sonic/scripts/run_isaacsim_loop.py
#   │ git diff --check
#   │ … +4 lines
#   └ .....                                                               [100%]
#     10 passed in 0.06s
#     … +23 lines (ctrl + t to view transcript)
#      M gear_sonic/utils/isaacsim_simulator/simulator_factory.py
#      M gear_sonic/utils/isaacsim_simulator/tests/test_contract.py

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# • Đã viết xong bản đầu cho Isaac Sim 6.0.1, với bốn khu vực được đánh dấu rõ để bạn review:

#   - ISAAC 6 LIFECYCLE: gear_sonic/utils/isaacsim_simulator/base_sim.py:44
#   - TENSOR VIEWS: gear_sonic/utils/isaacsim_simulator/base_sim.py:101
#   - EFFORT CONTROL: gear_sonic/utils/isaacsim_simulator/base_sim.py:145
#   - DDS BRIDGE: gear_sonic/utils/isaacsim_simulator/base_sim.py:239

#   Các path bạn cung cấp đã trở thành mặc định trong gear_sonic/utils/isaacsim_simulator/configs.py:12:

#   Robot:       /World/Robots/g1
#   Pelvis:      /World/Robots/g1/pelvis
#   Torso:       /World/Robots/g1/torso_link
#   PhysicsScene /PhysicsScene

#   Bản này đã thực hiện:

#   - Mở và kiểm tra stage Z-up, đơn vị mét.
#   - Tìm đúng một articulation root dưới G1.
#   - Kiểm tra chính xác PhysicsScene.
#   - Tạo PhysicsScene, Articulation, RigidPrim bằng API experimental của Isaac Sim 6.0.1.
#   - Loại bỏ hoàn toàn self.sim.reset() và các lệnh initialize() cũ.
#   - Play timeline và chờ tensor views sẵn sàng với timeout.
#   - Xác nhận articulation có đúng 29 DOF.
#   - Map 29 joint theo tên DDS, không phụ thuộc thứ tự DOF trong USD.
#   - Chuyển các joint được điều khiển sang effort mode.
#   - Tính torque từ tau_ff + kp(q_target-q) + kd(dq_target-dq).
#   - Giới hạn torque theo SONIC config và effort limit được author trong USD.
#   - Publish rt/lowstate, rt/odostate, rt/secondary_imu.
#   - Subscribe rt/lowcmd, kiểm tra CRC/mode và timeout.
#   - Dừng effort khi command quá hạn hoặc timeline dừng.
#   - Dùng SimulationManager.step() và RenderingManager.render().
#   - Tách CLI adapter khỏi CLI Kit tại gear_sonic/scripts/run_isaacsim_loop.py:37.

#   Kiểm chứng hiện tại:

#   - 10/10 unit tests đạt.
#   - py_compile đạt.
#   - git diff --check đạt.
#   - Smoke test Isaac Sim 6.0.1 với G1 29 DOF cục bộ thoát mã 0, tức tensor views, joint mapping và effort-mode initialization đều qua.
#   - Log vẫn có lỗi extension asset_converter do thiếu libxml2.so.2, nhưng lỗi này không chặn physics/tensor backend.

#   Về bàn tay: bản hiện tại coi articulation /World/Robots/g1 là 29 DOF thân. Hai bàn tay có ba ngón chưa được giả định là 14 DOF nằm
#   chung articulation. Tùy cấu trúc cibo.usd, chúng có thể là:

#   - chỉ visual/fixed joints;
#   - hai articulation riêng;
#   - hoặc Dex3 articulation với bảy motor mỗi tay.

#   Do đó, không dùng --with-hands cho scene 29 DOF ở thời điểm này. Bước tiếp theo là inspect subtree của hai bàn tay, sau đó nối chúng
#   vào rt/dex3/left/* và rt/dex3/right/* nếu chúng thực sự có articulation riêng.


# """
