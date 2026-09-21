"""CPU-only contract checks. Run with unittest; no Isaac Sim or DDS installation needed."""
import copy
import ast
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from gear_sonic.utils.isaacsim_simulator.control import (
    command_torques, joint_groups, map_joints, world_to_body,
)
from gear_sonic.scripts.run_isaacsim_loop import parse_args


def motor(q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0):
    return types.SimpleNamespace(q=q, dq=dq, kp=kp, kd=kd, tau=tau)


def message():
    return types.SimpleNamespace(
        motor_cmd=[motor() for _ in range(35)],
        motor_state=[types.SimpleNamespace(q=0., dq=0., ddq=0., tau_est=0.) for _ in range(35)],
        imu_state=types.SimpleNamespace(quaternion=[0.]*4, gyroscope=[0.]*3, accelerometer=[0.]*3),
        quaternion=[0.]*4, gyroscope=[0.]*3, accelerometer=[0.]*3,
        position=[0.]*3, linear_velocity=[0.]*3, angular_velocity=[0.]*3,
        orientation=[0.]*4, tick=0, crc=123, mode_pr=0,
    )


class ControlTests(unittest.TestCase):
    def test_joint_contract_matches_existing_robot_source(self):
        source = Path(__file__).resolve().parents[3] / "data/robot_model/supplemental_info/g1/g1_supplemental_info.py"
        assignments = {}
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ("body_actuated_joints", "left_hand_actuated_joints", "right_hand_actuated_joints"):
                    assignments[name] = ast.literal_eval(node.value)
        for name, joints in joint_groups().items():
            self.assertEqual(list(joints), assignments[name+"_actuated_joints"])

    def test_mapping_uses_names_not_usd_order(self):
        groups = joint_groups()
        ordered = sum((list(v) for v in groups.values()), [])
        shuffled = ordered[::-1]
        mapping = map_joints(shuffled, groups)
        for group, names in groups.items():
            self.assertEqual([shuffled[i] for i in mapping[group]], list(names))
        self.assertEqual(len(mapping["body"]), 29)
        self.assertEqual(len(mapping["left_hand"]), 7)
        with self.assertRaises(ValueError):
            map_joints(shuffled[:-1], groups)
        with self.assertRaises(ValueError):
            map_joints(shuffled + [shuffled[0]], groups)

    def test_pd_feedforward_limits_and_sentinels(self):
        commands = [motor(q=1, kp=10, dq=2, kd=2, tau=3),
                    motor(q=2146000000., kp=50, dq=16000, kd=5, tau=2)]
        actual = command_torques(commands, np.array([0., 1.]), np.array([1., 0.]),
                                 np.array([8., 3.]))
        np.testing.assert_allclose(actual, [8., 2.])
        commands[0].q = float("nan")
        with self.assertRaises(ValueError):
            command_torques(commands, np.zeros(2), np.zeros(2), np.ones(2))

    def test_gyro_frame_and_quaternion_order(self):
        q = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        np.testing.assert_allclose(world_to_body(q, [1, 0, 0]), [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(world_to_body([1, 0, 0, 0], [0, 0, 9.81]), [0, 0, 9.81])
        with self.assertRaises(ValueError):
            world_to_body([0, 0, 0, 0], [0, 0, 1])

    def test_cli_validation_without_kit(self):
        with tempfile.NamedTemporaryFile(suffix=".usd") as scene:
            cfg = parse_args(["--usd-path", scene.name, "--robot-path", "/World/G1", "--inspect"])
            self.assertEqual(cfg.render_every, 20)
            self.assertEqual(cfg.render_fps, 25.0)
            video = parse_args([
                "--usd-path", scene.name,
                "--record-video", "capture.mp4",
                "--video-fps", "20",
            ])
            self.assertTrue(video.record_video.endswith("capture.mp4"))
            self.assertEqual(video.video_fps, 20.0)
            self.assertTrue(cfg.inspect)
            self.assertFalse(cfg.inspect_only)
            self.assertEqual(cfg.domain_id, 0)
            self.assertIsNone(cfg.with_hands)
            inspect_only = parse_args(["--usd-path", scene.name, "--inspect-only"])
            self.assertTrue(inspect_only.inspect_only)
            self.assertFalse(inspect_only.inspect)
            self.assertTrue(parse_args(["--usd-path", scene.name, "--with-hands"]).with_hands)
            self.assertFalse(parse_args(["--usd-path", scene.name, "--without-hands"]).with_hands)
            with self.assertRaises(ValueError):
                parse_args(["--usd-path", scene.name, "--robot-path", "/World/G1",
                            "--physics-dt", "nan"])


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.channels = []
        channels = self.channels
        class FakeChannel:
            def __init__(self, name, kind):
                self.name = name
                self.closed = False
                self.samples = []
                channels.append(self)
            def Init(self, handler=None, *args):
                # Force the initialization race found in the copied bridge
                if handler is not None:
                    handler(message())
            def Write(self, sample):
                self.samples.append(copy.deepcopy(sample))
            def Close(self):
                self.closed = True
        modules = {}
        def module(name, **attrs):
            result = types.ModuleType(name)
            result.__dict__.update(attrs)
            modules[name] = result
            return result
        module("unitree_sdk2py.core.channel", ChannelPublisher=FakeChannel, ChannelSubscriber=FakeChannel)
        module("unitree_sdk2py.utils.crc", CRC=lambda: types.SimpleNamespace(Crc=lambda msg: 123))
        defaults = {name: message for name in (
            "unitree_go_msg_dds__WirelessController_", "unitree_hg_msg_dds__HandCmd_",
            "unitree_hg_msg_dds__HandState_", "unitree_hg_msg_dds__IMUState_",
            "unitree_hg_msg_dds__LowCmd_", "unitree_hg_msg_dds__LowState_",
            "unitree_hg_msg_dds__OdoState_")}
        module("unitree_sdk2py.idl.default", **defaults)
        module("unitree_sdk2py.idl.unitree_go.msg.dds_", WirelessController_=object)
        module("unitree_sdk2py.idl.unitree_hg.msg.dds_",
               **{name: object for name in ("HandCmd_", "HandState_", "OdoState_",
                                           "IMUState_", "LowCmd_", "LowState_")})
        self.patch = patch.dict(sys.modules, modules)
        self.patch.start()
        name = "gear_sonic.utils.isaacsim_simulator.unitree_sdk2py_bridge"
        sys.modules.pop(name, None)
        self.module_name = name
        cls = importlib.import_module(name).UnitreeSdk2Bridge
        self.bridge = cls({"ROBOT_TYPE": "g1_29dof", "NUM_MOTORS": 29,
                           "NUM_HAND_MOTORS": 7, "USE_SENSOR": False})

    def tearDown(self):
        self.bridge.close()
        self.patch.stop()
        sys.modules.pop(self.module_name, None)

    def test_callback_at_initialization_and_snapshot_isolation(self):
        first = self.bridge.command_snapshot()
        self.assertIsNotNone(first["body"][1])
        first["body"][0].motor_cmd[0].q = 99
        self.assertEqual(self.bridge.command_snapshot()["body"][0].motor_cmd[0].q, 0)
        bad = message()
        bad.crc = 0
        before = self.bridge.command_snapshot()["body"][1]
        self.bridge.LowCmdHandler(bad)
        self.assertEqual(before, self.bridge.command_snapshot()["body"][1])
        bad.crc = 123
        bad.mode_pr = 1
        self.bridge.LowCmdHandler(bad)
        self.assertEqual(before, self.bridge.command_snapshot()["body"][1])
        stats = self.bridge.command_stats()
        self.assertEqual(stats["received"]["body"], 1)
        self.assertEqual(stats["rejected_body_crc"], 1)
        self.assertEqual(stats["rejected_body_mode"], 1)
        self.bridge.reset()
        self.assertIsNone(self.bridge.command_snapshot()["body"][1])

    def test_publish_contract_crc_secondary_imu_and_hands(self):
        obs = dict(time=.02, floating_base_pose=[1, 2, 3, 1, 0, 0, 0],
                   floating_base_vel=[1, 2, 3, 4, 5, 6],
                   floating_base_acc=[0, 0, 9.81, 0, 0, 0],
                   secondary_imu_quat=[1, 0, 0, 0],
                   secondary_imu_vel=[0, 0, 0, 7, 8, 9],
                   odometry_angular_velocity=[10, 11, 12])
        for group, count in (("body", 29), ("left_hand", 7), ("right_hand", 7)):
            obs[group+"_q"] = list(range(count))
            obs[group+"_dq"] = [0.]*count
        obs["body_ddq"] = [0.]*29
        obs["body_tau_est"] = [0.]*29
        self.bridge.PublishLowState(obs)
        published = {c.name: c.samples[-1] for c in self.channels if c.samples}
        self.assertEqual(published["rt/lowstate"].crc, 123)
        self.assertEqual(published["rt/lowstate"].tick, 20)
        self.assertEqual(published["rt/lowstate"].motor_state[28].q, 28)
        self.assertEqual(published["rt/secondary_imu"].gyroscope, [7, 8, 9])
        self.assertEqual(published["rt/odostate"].angular_velocity, [10, 11, 12])
        self.assertEqual(published["rt/dex3/right/state"].motor_state[6].q, 6)
        self.bridge.close()
        self.assertTrue(all(c.closed for c in self.channels))


class LoopTests(unittest.TestCase):
    def make_sim(self):
        from gear_sonic.utils.isaacsim_simulator.base_sim import BaseSimulator
        sim = BaseSimulator.__new__(BaseSimulator)
        sim.config = types.SimpleNamespace(inspect=False, inspect_only=False,
                                          with_hands=True, physics_dt=.002,
                                          render_every=10, command_timeout=.5,
                                          headless=True, realtime=False)
        sim.app = Mock()
        sim.app.is_running.side_effect = [True, False]
        sim.timeline = Mock()
        sim.timeline.is_playing.return_value = True
        sim._SimulationManager = Mock()
        sim._RenderingManager = Mock()
        sim._observation = Mock(return_value={})
        sim.step_count = 0
        sim._latest_q = np.zeros(1)
        sim._latest_dq = np.zeros(1)
        sim._render_enabled = False
        sim._render_every = 20
        sim._video_recorder = None
        return sim

    def test_video_capture_uses_simulation_time_without_catchup(self):
        sim = self.make_sim()
        sim.config.video_fps = 20.0
        sim.config.video_width = 4
        sim.config.video_height = 3
        sim._next_video_time = 0.05
        sim._rgb_annotator = Mock(
            get_data=Mock(return_value=np.zeros((3, 4, 4), dtype=np.uint8))
        )
        sim._video_recorder = Mock()

        sim._capture_video_frame(0.049)
        sim._video_recorder.submit.assert_not_called()
        sim._capture_video_frame(0.05)
        sim._capture_video_frame(0.251)

        self.assertEqual(sim._video_recorder.submit.call_count, 2)
        self.assertAlmostEqual(sim._next_video_time, 0.30)
        self.assertEqual(sim._video_recorder.submit.call_args.args[0].shape, (3, 4, 3))

    def test_waiting_for_first_command_does_not_step_physics(self):
        sim = self.make_sim()
        bridge = Mock()
        bridge.command_snapshot.return_value = {"body": (None, None)}
        fake = types.ModuleType("bridge")
        fake.UnitreeSdk2Bridge = Mock(return_value=bridge)
        with patch.dict(sys.modules, {
            "gear_sonic.utils.isaacsim_simulator.unitree_sdk2py_bridge": fake
        }), patch("gear_sonic.utils.isaacsim_simulator.simulator_factory.init_channel"), patch("time.sleep"):
            sim.start()
        sim._SimulationManager.step.assert_not_called()
        bridge.PublishLowState.assert_called_once()

    def test_stale_body_command_stops_before_physics_step(self):
        sim = self.make_sim()
        bridge = Mock()
        bridge.command_snapshot.return_value = {"body": (None, 0.)}
        fake = types.ModuleType("bridge")
        fake.UnitreeSdk2Bridge = Mock(return_value=bridge)
        with patch.dict(sys.modules, {
            "gear_sonic.utils.isaacsim_simulator.unitree_sdk2py_bridge": fake
        }), patch("gear_sonic.utils.isaacsim_simulator.simulator_factory.init_channel"):
            with self.assertRaises(TimeoutError):
                sim.start()
        sim._SimulationManager.step.assert_not_called()

    def test_valid_command_switches_to_effort_and_steps_once(self):
        import time
        sim = self.make_sim()
        sim.config.realtime = False
        sim.step_count = 0
        sim.indices = {"body": np.array([0])}
        sim.limits = {"body": np.array([2.])}
        sim._last_torque = np.zeros(1)
        sim.robot = Mock()
        sim.robot.get_dof_positions.return_value = np.zeros((1, 1))
        sim.robot.get_dof_velocities.return_value = np.zeros((1, 1))
        cmd = message()
        cmd.motor_cmd[0] = motor(q=1, kp=10)
        cmd.motor_cmd[0].mode = 1
        bridge = Mock()
        bridge.command_snapshot.return_value = {"body": (cmd, time.monotonic())}
        fake = types.ModuleType("bridge")
        fake.UnitreeSdk2Bridge = Mock(return_value=bridge)
        with patch.dict(sys.modules, {
            "gear_sonic.utils.isaacsim_simulator.unitree_sdk2py_bridge": fake
        }), patch("gear_sonic.utils.isaacsim_simulator.simulator_factory.init_channel"):
            sim.start()
        np.testing.assert_allclose(sim.robot.set_dof_efforts.call_args.args[0], [[2.]])
        sim._SimulationManager.step.assert_called_once_with()
        self.assertEqual(sim.step_count, 1)


if __name__ == "__main__":
    unittest.main()
