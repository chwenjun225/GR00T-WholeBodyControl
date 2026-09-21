"""Isaac Sim 6.0.1 adapter for SONIC Whole-Body Control over Unitree DDS."""

import contextlib
from pathlib import Path
import time

import numpy as np
import yaml  # type: ignore

from .control import command_torques
from .control import joint_groups
from .control import map_joints
from .control import world_to_body
from .profiling import TimingProfiler
from .state_reader import CppStateReader


def _numpy(value) -> np.ndarray:
    """Convert Warp/Torch/NumPy values returned by Isaac tensor views."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    elif hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class BaseSimulator:
    def __init__(self, config, simulation_app):
        # Omniverse modules must only be imported after SimulationApp is created
        import omni.timeline
        import omni.usd

        from pxr import Usd
        from pxr import UsdGeom
        from pxr import UsdPhysics

        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.core.experimental.utils.stage import is_stage_loading
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import PhysicsScene
        from isaacsim.core.simulation_manager import SimulationManager

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
        self._rgb_annotator = None
        self._render_product = None
        self._video_recorder = None
        self._next_video_time = config.video_fps and 1.0 / config.video_fps
        self._profiler = TimingProfiler() if config.profile else None
        self._profiling_active = False
        self._state_reader = None

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
        # prim wrappers do not use the legacy initialize()/reset() methods
        SimulationManager.setup_simulation(dt=config.physics_dt, device="cpu")
        self.physics_scene = PhysicsScene(scene_prim)
        self.physics_scene.set_dt(config.physics_dt)
        gui_render_fps = (1.0 / (config.physics_dt * config.render_every)
                          if not config.headless else 0.0)
        target_render_fps = max(
            gui_render_fps,
            config.video_fps if config.record_video else 0.0,
        )
        self._render_enabled = target_render_fps > 0
        self._render_every = (
            max(1, round(1.0 / (config.physics_dt * target_render_fps)))
            if self._render_enabled else config.render_every
        )
        RenderingManager.set_dt(config.physics_dt * self._render_every)
        if self._render_enabled:
            actual_render_fps = 1.0 / (config.physics_dt * self._render_every)
            print(
                f"[IsaacSim] Physics/DDS={1.0 / config.physics_dt:.1f} Hz; "
                f"render={actual_render_fps:.1f} FPS"
            )
        self.robot = Articulation(articulation_path)
        # Keep pelvis and torso in one physics view so both links can be fetched
        # by one batched C++ data-view update.
        self.rigid_bodies = RigidPrim([base_path, torso_path])

        # Playing the timeline and pumping one app update creates PhysX tensor views
        self.timeline.play()
        tensor_deadline = time.monotonic() + 10.0
        last_tensor_error = None
        while True:
            try:
                self.app.update()
                q = _numpy(self.robot.get_dof_positions())
                rigid_poses = tuple(
                    _numpy(value) for value in self.rigid_bodies.get_world_poses()
                )
                if q.ndim == 2 and q.shape[0] == 1:
                    break
            except Exception as exc:  # tensor view may not exist on the first frame
                last_tensor_error = exc
            if time.monotonic() > tensor_deadline:
                detail = last_tensor_error or "invalid tensor shape"
                raise RuntimeError(f"Isaac tensor views did not initialize: {detail}")

        if any(value.shape[0] != 2 for value in rigid_poses):
            raise RuntimeError("Pelvis/torso RigidPrim must resolve exactly two rigid bodies")

        self.dof_names = list(self.robot.dof_names)
        dof_count = len(self.dof_names)
        if q.shape != (1, dof_count) or dof_count not in (29, 43):
            raise ValueError(
                "Expected a supported G1 articulation with 29 body DOFs or "
                f"43 body+Dex3 DOFs, got names={dof_count}, positions={q.shape}."
            )

        if config.use_cpp_data_view:
            try:
                # NVIDIA's prim-data PhysX tests settle the timeline for several
                # updates before creating/reinitializing native reader views.
                for _ in range(5):
                    self.app.update()
                q = _numpy(self.robot.get_dof_positions())
                rigid_poses = tuple(
                    _numpy(value) for value in self.rigid_bodies.get_world_poses()
                )
                reader = CppStateReader(self.rigid_bodies)
                state = reader.read()
                if not np.allclose(state.positions, rigid_poses[0], rtol=1e-5, atol=1e-6):
                    raise RuntimeError("C++ rigid-body positions do not match the tensor view")
                quaternion_alignment = np.abs(
                    np.sum(state.orientations * rigid_poses[1], axis=1)
                )
                if not np.allclose(quaternion_alignment, 1.0, rtol=1e-4, atol=1e-4):
                    raise RuntimeError("C++ rigid-body quaternion order is not wxyz")
                self._state_reader = reader
                print("[IsaacSim] State reader: batched C++ RigidBodyDataView")
            except Exception as exc:
                print(f"[IsaacSim] C++ state reader unavailable; using tensor getters: {exc}")

        self._gravity_world = _numpy(self.physics_scene.get_gravity()).astype(np.float64)
        if self._gravity_world.shape != (3,) or not np.isfinite(self._gravity_world).all():
            raise RuntimeError(f"Invalid PhysicsScene gravity: {self._gravity_world}")
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

        if config.record_video:
            self._setup_video_recording()

    def _setup_video_recording(self):
        import omni.replicator.core as rep
        from pxr import UsdGeom

        camera = self.stage.GetPrimAtPath(self.config.video_camera_path)
        if not camera.IsValid() or not camera.IsA(UsdGeom.Camera):
            cameras = [str(prim.GetPath()) for prim in self.stage.Traverse()
                       if prim.IsA(UsdGeom.Camera)]
            raise ValueError(
                f"Video camera not found: {self.config.video_camera_path}; "
                f"available: {cameras}"
            )

        self._render_product = rep.create.render_product(
            self.config.video_camera_path,
            (self.config.video_width, self.config.video_height),
        )
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._rgb_annotator.attach([self._render_product])

        from .video_recorder import AsyncVideoRecorder

        self._video_recorder = AsyncVideoRecorder(
            self.config.record_video,
            self.config.video_fps,
            self.config.video_width,
            self.config.video_height,
        )
        print(
            f"[video] capture={self.config.video_fps:.1f} FPS "
            f"camera={self.config.video_camera_path} output={self.config.record_video}"
        )

    def _capture_video_frame(self, simulation_time: float):
        if self._video_recorder is None or simulation_time < self._next_video_time:
            return False

        rgb_data = self._rgb_annotator.get_data() # TODO: Render/capture vẫn chạy trong cùng process Isaac
        if isinstance(rgb_data, dict):
            rgb_data = rgb_data.get("data", np.array([], dtype=np.uint8))
        frame = np.asarray(rgb_data, dtype=np.uint8)
        if frame.size:
            if frame.ndim == 1:
                frame = frame.reshape(
                    self.config.video_height, self.config.video_width, -1
                )
            self._video_recorder.submit(frame[:, :, :3])

        video_dt = 1.0 / self.config.video_fps
        while self._next_video_time <= simulation_time:
            self._next_video_time += video_dt
        return True

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
        # TODO: Nghi vấn gây ra chậm trễ , chatpgt nhận xét "Nếu có body + 2 hands thì có thể thành 3 lần gọi Isaac API mỗi 2 ms. Nên cân nhắc gom torque toàn robot rồi gọi 1 lần duy nhất."
        for group, indices in self.indices.items():
            authored = usd_limits[indices]
            valid = np.isfinite(authored) & (authored > 0)
            limits[group] = np.where(valid, np.minimum(reference[group], authored),
                                     reference[group])
        return limits

    def _observation(self):
        profiler = self._profiler if self._profiling_active else None
        observation_start = time.perf_counter_ns() if profiler else 0

        # The Python articulation getters are already direct tensor reads and
        # measured at ~0.05 ms combined. Isaac 6.0.1's native articulation
        # prim-data getter currently segfaults, so only rigid-link state uses
        # the C++ batch path.
        started = time.perf_counter_ns() if profiler else 0
        q = _numpy(self.robot.get_dof_positions())[0].astype(np.float64)
        if profiler:
            profiler.add("dof_q", time.perf_counter_ns() - started)

        started = time.perf_counter_ns() if profiler else 0
        dq = _numpy(self.robot.get_dof_velocities())[0].astype(np.float64)
        if profiler:
            profiler.add("dof_dq", time.perf_counter_ns() - started)

        if self._state_reader is not None:
            started = time.perf_counter_ns() if profiler else 0
            state = self._state_reader.read()
            if profiler:
                profiler.add("state_update", time.perf_counter_ns() - started)
            positions = state.positions
            orientations = state.orientations
            linear = state.linear_velocities
            angular = state.angular_velocities
        else:
            started = time.perf_counter_ns() if profiler else 0
            positions, orientations = self.rigid_bodies.get_world_poses()
            if profiler:
                profiler.add("rigid_pose", time.perf_counter_ns() - started)

            started = time.perf_counter_ns() if profiler else 0
            linear, angular = self.rigid_bodies.get_velocities()
            if profiler:
                profiler.add("rigid_vel", time.perf_counter_ns() - started)

            positions = _numpy(positions)
            orientations = _numpy(orientations)
            linear = _numpy(linear)
            angular = _numpy(angular)

        self._latest_q = q
        self._latest_dq = dq

        started = time.perf_counter_ns() if profiler else 0
        p = positions[0]
        quat = orientations[0]
        torso_quat = orientations[1]
        velocity = linear[0]
        base_angular = angular[0]
        torso_angular = angular[1]
        arrays = (q, dq, p, quat, torso_quat, velocity, base_angular, torso_angular)
        if not all(np.isfinite(value).all() for value in arrays):
            raise RuntimeError("Non-finite simulated robot state")

        acceleration = (np.zeros(3) if self._last_velocity is None else
                        (velocity - self._last_velocity) / self.config.physics_dt)
        ddq = (np.zeros_like(dq) if self._last_dq is None else
               (dq - self._last_dq) / self.config.physics_dt)
        if profiler:
            profiler.add("obs_numpy", time.perf_counter_ns() - started)

        started = time.perf_counter_ns() if profiler else 0
        obs = {
            "time": self._SimulationManager.get_simulation_time(),
            "floating_base_pose": np.r_[p, quat],
            "floating_base_vel": np.r_[velocity, world_to_body(quat, base_angular)],
            "floating_base_acc": np.r_[world_to_body(
                                            quat, acceleration - self._gravity_world),
                                        np.zeros(3)],
            "odometry_angular_velocity": base_angular,
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

        if profiler:
            profiler.add("obs_pack", time.perf_counter_ns() - started)
            profiler.add("observation", time.perf_counter_ns() - observation_start)

        return obs

    def start(self):
        if self.config.inspect or self.config.inspect_only:
            print("[IsaacSim] Running observation test...")

            for i in range(10):
                self._SimulationManager.step() # TODO: Đây là nghi phạm lớn thứ hai. Nếu riêng hàm này đã tốn ~8–10 ms thì bottleneck là PhysX / scene complexity, không phải Python bridge.

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
        report_time = time.monotonic()
        report_step = self.step_count
        report_commands = 0
        compute_time = 0.0
        max_compute_time = 0.0
        profiler = self._profiler
        while self.app.is_running():
            compute_start = time.monotonic()
            compute_start_ns = time.perf_counter_ns() if self._profiling_active else 0
            observation = self._observation()

            started = time.perf_counter_ns() if self._profiling_active else 0
            self.bridge.PublishLowState(observation) # TODO: DDS publish mỗi step, 500 lần/giây. Có thể không quá nặng, nhưng vẫn cần profile riêng
            if self._profiling_active:
                profiler.add("dds", time.perf_counter_ns() - started)
            if not state_publish_reported:
                print(
                    "DDS state published: rt/lowstate, rt/odostate, "
                    "rt/secondary_imu, rt/dex3/{left,right}/state."
                )
                state_publish_reported = True

            started = time.perf_counter_ns() if self._profiling_active else 0
            snapshot = self.bridge.command_snapshot()
            if self._profiling_active:
                profiler.add("snapshot", time.perf_counter_ns() - started)
            if snapshot["body"][1] is None:
                if self._render_enabled and waiting_frames % self._render_every == 0:
                    self._RenderingManager.render() # TODO: Render/capture vẫn chạy trong cùng process Isaac
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
                report_time = now
                report_step = self.step_count
                report_commands = self.bridge.command_stats()["received"]["body"]
                compute_time = 0.0
                max_compute_time = 0.0
                if profiler is not None:
                    profiler.reset()
                    self._profiling_active = True
                    compute_start_ns = time.perf_counter_ns()

            q = self._latest_q
            dq = self._latest_dq
            torque_elapsed_ns = 0
            self._last_torque.fill(0.0)
            for group, indices in self.indices.items():
                started = time.perf_counter_ns() if self._profiling_active else 0
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
                if self._profiling_active:
                    torque_elapsed_ns += time.perf_counter_ns() - started
                self._last_torque[indices] = torque

            # One tensor write for the complete articulation, matching MuJoCo's
            # single mj_data.ctrl assignment instead of one API call per group.
            started = time.perf_counter_ns() if self._profiling_active else 0
            self.robot.set_dof_efforts(self._last_torque[np.newaxis, :])
            if self._profiling_active:
                profiler.add("torque", torque_elapsed_ns)
                profiler.add("set_effort", time.perf_counter_ns() - started)

            started = time.perf_counter_ns() if self._profiling_active else 0
            self._SimulationManager.step()
            if self._profiling_active:
                profiler.add("physics", time.perf_counter_ns() - started)
            self.step_count += 1
            if self._render_enabled and self.step_count % self._render_every == 0:
                started = time.perf_counter_ns() if self._profiling_active else 0
                self._RenderingManager.render()
                if self._profiling_active:
                    profiler.add("render", time.perf_counter_ns() - started)

                started = time.perf_counter_ns() if self._profiling_active else 0
                captured = self._capture_video_frame(
                    self._SimulationManager.get_simulation_time()
                )
                if self._profiling_active and captured:
                    profiler.add("capture", time.perf_counter_ns() - started)

            compute_elapsed = time.monotonic() - compute_start
            compute_time += compute_elapsed
            max_compute_time = max(max_compute_time, compute_elapsed)
            if self._profiling_active:
                profiler.add("loop", time.perf_counter_ns() - compute_start_ns)

            report_now = time.monotonic()
            if report_now - report_time >= 1.0:
                wall_dt = report_now - report_time
                steps = self.step_count - report_step
                stats = self.bridge.command_stats()
                commands = stats["received"]["body"]
                command_hz = (commands - report_commands) / wall_dt
                command_at = stats["last_received_at"]["body"]
                command_age_ms = ((report_now - command_at) * 1e3
                                  if command_at is not None else float("inf"))
                physics_hz = steps / wall_dt
                mean_compute_ms = compute_time * 1e3 / max(steps, 1)
                body_tau = self._last_torque[self.indices["body"]]
                video_status = ""
                if self._video_recorder is not None:
                    video = self._video_recorder.stats()
                    video_status = (
                        f" video={video['submitted']} dropped={video['dropped']}"
                    )
                print(
                    f"[control] physics={physics_hz:.1f} Hz "
                    f"rtf={physics_hz * self.config.physics_dt:.2f} "
                    f"compute={mean_compute_ms:.2f}/{max_compute_time * 1e3:.2f} ms(avg/max) "
                    f"lowcmd={command_hz:.1f} Hz age={command_age_ms:.2f} ms "
                    f"|tau|max={np.max(np.abs(body_tau)):.2f} Nm "
                    f"reject_crc={stats['rejected_body_crc']} "
                    f"reject_mode={stats['rejected_body_mode']}"
                    f"{video_status}"
                )
                if profiler is not None:
                    print(
                        "[profile] avg/max per call: "
                        + profiler.format((
                            "observation", "dds", "snapshot", "torque",
                            "set_effort", "physics", "render", "capture", "loop",
                        ))
                    )
                    print(
                        "[profile.obs] avg/max per call: "
                        + profiler.format((
                            "state_update", "dof_q", "dof_dq", "rigid_pose",
                            "rigid_vel", "obs_numpy", "obs_pack",
                        ))
                    )
                    profiler.reset()
                report_time = report_now
                report_step = self.step_count
                report_commands = commands
                compute_time = 0.0
                max_compute_time = 0.0

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
        dof_count = len(self._last_torque) if self._last_torque is not None else 0
        if dof_count:
            self.robot.set_dof_efforts(np.zeros((1, dof_count)))
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
            if self._rgb_annotator is not None and self._render_product is not None:
                with contextlib.suppress(Exception):
                    self._rgb_annotator.detach([self._render_product])
            if self._render_product is not None:
                with contextlib.suppress(Exception):
                    self._render_product.destroy()
            if self._video_recorder is not None:
                self._video_recorder.close()
                print(f"[video] saved: {self.config.record_video}")
            if self.bridge is not None:
                self.bridge.close()
