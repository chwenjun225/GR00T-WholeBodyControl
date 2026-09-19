"""Unitree DDS transport for the Isaac Sim backend (no physics API calls).

Subscribes to low-level motor commands (body + hands) and publishes
simulated sensor state (joint pos/vel, IMU, odometry) back over DDS,
so the WBC policy sees the sim as a real robot.
"""

import sys
import copy
import time
import threading
from typing import Dict, Tuple

import numpy as np
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__WirelessController_,
    unitree_hg_msg_dds__HandCmd_ as HandCmd_default,
    unitree_hg_msg_dds__HandState_ as HandState_default,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, OdoState_


class UnitreeSdk2Bridge:
    """
    This class is responsible for bridging the Unitree SDK2 with the Groot environment.
    It is responsible for sending and receiving messages to and from the Unitree SDK2.
    Both the body and hand are supported.
    """

    def __init__(self, config):
        # A reader may invoke its callback during Init: initialize all shared state first.
        self.low_cmd_lock = threading.Lock()
        self.left_hand_cmd_lock = threading.Lock()
        self.right_hand_cmd_lock = threading.Lock()
        self._received_at = {"body": None, "left_hand": None, "right_hand": None}
        self._crc = CRC()
        self.reset()
        # Note that we do not give the mjdata and mjmodel to the UnitreeSdk2Bridge.
        # It is unsafe and would be unflexible if we use a hand-plugged robot model

        robot_type = config["ROBOT_TYPE"]
        if "g1" in robot_type or "h1-2" in robot_type:
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__IMUState_ as IMUState_default,
                unitree_hg_msg_dds__LowCmd_,
                unitree_hg_msg_dds__LowState_ as LowState_default,
                unitree_hg_msg_dds__OdoState_ as OdoState_default,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif "h1" == robot_type or "go2" == robot_type:
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__LowCmd_,
                unitree_go_msg_dds__LowState_ as LowState_default,
                unitree_hg_msg_dds__IMUState_ as IMUState_default,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_, LowCmd_, LowState_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        else:
            raise ValueError(f"Invalid robot type '{robot_type}'. Expected 'g1', 'h1', or 'go2'.")

        self.num_body_motor = config["NUM_MOTORS"]
        self.num_hand_motor = config.get("NUM_HAND_MOTORS", 0)
        self.use_sensor = config["USE_SENSOR"]

        self.have_imu_ = False
        self.have_frame_sensor_ = False

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        # Only create odo_state for supported robot types
        if "g1" in robot_type or "h1-2" in robot_type:
            self.odo_state = OdoState_default()
            self.odo_state_puber = ChannelPublisher("rt/odostate", OdoState_)
            self.odo_state_puber.Init()
        else:
            self.odo_state = None
            self.odo_state_puber = None
        self.torso_imu_state = IMUState_default()
        self.torso_imu_puber = ChannelPublisher("rt/secondary_imu", IMUState_)
        self.torso_imu_puber.Init()

        self.left_hand_state = HandState_default()
        self.left_hand_state_puber = ChannelPublisher("rt/dex3/left/state", HandState_)
        self.left_hand_state_puber.Init()
        self.right_hand_state = HandState_default()
        self.right_hand_state_puber = ChannelPublisher("rt/dex3/right/state", HandState_)
        self.right_hand_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 1)

        self.left_hand_cmd = HandCmd_default()
        self.left_hand_cmd_suber = ChannelSubscriber("rt/dex3/left/cmd", HandCmd_)
        self.left_hand_cmd_suber.Init(self.LeftHandCmdHandler, 1)
        self.right_hand_cmd = HandCmd_default()
        self.right_hand_cmd_suber = ChannelSubscriber("rt/dex3/right/cmd", HandCmd_)
        self.right_hand_cmd_suber.Init(self.RightHandCmdHandler, 1)


        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            "rt/wirelesscontroller", WirelessController_
        )
        self.wireless_controller_puber.Init()

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }
        self.joystick = None


    def reset(self):
        with self.low_cmd_lock:
            self.low_cmd_received = False
            self.new_low_cmd = False
            self._received_at["body"] = None
        with self.left_hand_cmd_lock:
            self.left_hand_cmd_received = False
            self.new_left_hand_cmd = False
            self._received_at["left_hand"] = None
        with self.right_hand_cmd_lock:
            self.right_hand_cmd_received = False
            self.new_right_hand_cmd = False
            self._received_at["right_hand"] = None

    def LowCmdHandler(self, msg):
        if self._crc.Crc(msg) != msg.crc or msg.mode_pr != 0:
            return
        with self.low_cmd_lock:
            self.low_cmd = msg
            self.low_cmd_received = True
            self.new_low_cmd = True
            self._received_at["body"] = time.monotonic()

    def LeftHandCmdHandler(self, msg):
        with self.left_hand_cmd_lock:
            self.left_hand_cmd = msg
            self.left_hand_cmd_received = True
            self.new_left_hand_cmd = True
            self._received_at["left_hand"] = time.monotonic()

    def RightHandCmdHandler(self, msg):
        with self.right_hand_cmd_lock:
            self.right_hand_cmd = msg
            self.right_hand_cmd_received = True
            self.new_right_hand_cmd = True
            self._received_at["right_hand"] = time.monotonic()

    def command_snapshot(self):
        """Copy commands under locks; never access PhysX from DDS callbacks."""
        with self.low_cmd_lock, self.left_hand_cmd_lock, self.right_hand_cmd_lock:
            return {
                "body": (copy.deepcopy(self.low_cmd), self._received_at["body"]),
                "left_hand": (copy.deepcopy(self.left_hand_cmd), self._received_at["left_hand"]),
                "right_hand": (copy.deepcopy(self.right_hand_cmd), self._received_at["right_hand"]),
            }

    def close(self):
        for name in ("low_cmd_suber", "left_hand_cmd_suber", "right_hand_cmd_suber",
                     "low_state_puber", "odo_state_puber", "torso_imu_puber",
                     "left_hand_state_puber", "right_hand_state_puber",
                     "wireless_controller_puber"):
            channel = getattr(self, name, None)
            if channel is not None:
                channel.Close()

    def cmd_received(self):
        with self.low_cmd_lock:
            low_cmd_received = self.low_cmd_received
        with self.left_hand_cmd_lock:
            left_hand_cmd_received = self.left_hand_cmd_received
        with self.right_hand_cmd_lock:
            right_hand_cmd_received = self.right_hand_cmd_received
        return low_cmd_received or left_hand_cmd_received or right_hand_cmd_received

    def PublishLowState(self, obs: Dict[str, any]):
        # publish body state
        if self.use_sensor:
            raise NotImplementedError("Sensor data is not implemented yet.")
        else:
            for i in range(self.num_body_motor):
                self.low_state.motor_state[i].q = obs["body_q"][i]
                self.low_state.motor_state[i].dq = obs["body_dq"][i]
                self.low_state.motor_state[i].ddq = obs["body_ddq"][i]
                self.low_state.motor_state[i].tau_est = obs["body_tau_est"][i]

        if self.use_sensor and self.have_frame_sensor_:
            raise NotImplementedError("Frame sensor data is not implemented yet.")
        else:
            # Get data from ground truth
            self.odo_state.position[:] = obs["floating_base_pose"][:3]
            self.odo_state.linear_velocity[:] = obs["floating_base_vel"][:3]
            self.odo_state.orientation[:] = obs["floating_base_pose"][3:7]
            self.odo_state.angular_velocity[:] = obs.get(
                "odometry_angular_velocity", obs["floating_base_vel"][3:6])
            # quaternion: w, x, y, z
            self.low_state.imu_state.quaternion[:] = obs["floating_base_pose"][3:7]
            # angular velocity
            self.low_state.imu_state.gyroscope[:] = obs["floating_base_vel"][3:6]
            # linear acceleration
            self.low_state.imu_state.accelerometer[:] = obs["floating_base_acc"][:3]

            self.torso_imu_state.quaternion[:] = obs["secondary_imu_quat"]
            self.torso_imu_state.gyroscope[:] = obs["secondary_imu_vel"][3:6]

        # acceleration: x, y, z (only available when frame sensor is enabled)
        if self.have_frame_sensor_:
            raise NotImplementedError("Frame sensor data is not implemented yet.")
        self.low_state.tick = int(obs["time"] * 1e3) & 0xffffffff
        self.low_state.crc = self._crc.Crc(self.low_state)
        self.low_state_puber.Write(self.low_state)

        self.odo_state.tick = int(obs["time"] * 1e3)
        self.odo_state_puber.Write(self.odo_state)

        self.torso_imu_puber.Write(self.torso_imu_state)

        # publish hand state
        for i in range(self.num_hand_motor):
            self.left_hand_state.motor_state[i].q = obs["left_hand_q"][i]
            self.left_hand_state.motor_state[i].dq = obs["left_hand_dq"][i]
        if self.num_hand_motor:
            self.left_hand_state_puber.Write(self.left_hand_state)

        for i in range(self.num_hand_motor):
            self.right_hand_state.motor_state[i].q = obs["right_hand_q"][i]
            self.right_hand_state.motor_state[i].dq = obs["right_hand_dq"][i]
        if self.num_hand_motor:
            self.right_hand_state_puber.Write(self.right_hand_state)

    def GetAction(self) -> Tuple[np.ndarray, bool, bool]:
        with self.low_cmd_lock:
            body_q = [self.low_cmd.motor_cmd[i].q for i in range(self.num_body_motor)]
        with self.left_hand_cmd_lock:
            left_hand_q = [self.left_hand_cmd.motor_cmd[i].q for i in range(self.num_hand_motor)]
        with self.right_hand_cmd_lock:
            right_hand_q = [self.right_hand_cmd.motor_cmd[i].q for i in range(self.num_hand_motor)]
        with self.low_cmd_lock, self.left_hand_cmd_lock, self.right_hand_cmd_lock:
            is_new_action = self.new_low_cmd and self.new_left_hand_cmd and self.new_right_hand_cmd
            if is_new_action:
                self.new_low_cmd = False
                self.new_left_hand_cmd = False
                self.new_right_hand_cmd = False

        return (
            np.concatenate([body_q[:-7], left_hand_q, body_q[-7:], right_hand_q]),
            self.cmd_received(),
            is_new_action,
        )

    def PublishWirelessController(self):
        import pygame

        if self.joystick is not None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(self.button_id["RB"])
            key_state[self.key_map["L1"]] = self.joystick.get_button(self.button_id["LB"])
            key_state[self.key_map["start"]] = self.joystick.get_button(self.button_id["START"])
            key_state[self.key_map["select"]] = self.joystick.get_button(self.button_id["SELECT"])
            key_state[self.key_map["R2"]] = self.joystick.get_axis(self.axis_id["RT"]) > 0
            key_state[self.key_map["L2"]] = self.joystick.get_axis(self.axis_id["LT"]) > 0
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            # Pack 16 button states into a single integer via bit-shifting
            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        import pygame

        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()

        if js_type == "xbox":
            if sys.platform.startswith("linux"):
                self.axis_id = {
                    "LX": 0, "LY": 1, "RX": 3, "RY": 4,
                    "LT": 2, "RT": 5, "DX": 6, "DY": 7,
                }
                self.button_id = {
                    "X": 2, "Y": 3, "B": 1, "A": 0,
                    "LB": 4, "RB": 5, "SELECT": 6, "START": 7,
                    "XBOX": 8, "LSB": 9, "RSB": 10,
                }
            elif sys.platform == "darwin":
                self.axis_id = {
                    "LX": 0, "LY": 1, "RX": 2, "RY": 3,
                    "LT": 4, "RT": 5,
                }
                self.button_id = {
                    "X": 2, "Y": 3, "B": 1, "A": 0,
                    "LB": 9, "RB": 10, "SELECT": 4, "START": 6,
                    "XBOX": 5, "LSB": 7, "RSB": 8,
                    "DYU": 11, "DYD": 12, "DXL": 13, "DXR": 14,
                }
            else:
                print("Unsupported OS. ")

        elif js_type == "switch":
            self.axis_id = {
                "LX": 0, "LY": 1, "RX": 2, "RY": 3,
                "LT": 5, "RT": 4, "DX": 6, "DY": 7,
            }
            self.button_id = {
                "X": 3, "Y": 4, "B": 1, "A": 0,
                "LB": 6, "RB": 7, "SELECT": 10, "START": 11,
            }
        else:
            print("Unsupported gamepad. ")
