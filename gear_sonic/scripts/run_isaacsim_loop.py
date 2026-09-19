"""Run the USD-to-Unitree DDS adapter using Isaac Sim's python.sh.

Example: ./python.sh -m gear_sonic.scripts.run_isaacsim_loop --help
"""
import tyro 
from typing import Dict 

from gear_sonic.utils.isaacsim_simulator.simulator_factory import init_channel
from gear_sonic.utils.isaacsim_simulator.simulator_factory import SimulatorFactory
from gear_sonic.utils.isaacsim_simulator.configs import SimLoopConfig
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model 
from gear_sonic.data.robot_model.robot_model import RobotModel 

ArgsConfig = SimLoopConfig 

class SimWrapper:
    def __init__(self, robot_model: RobotModel, env_name: str, config: Dict[str, any], **kwargs):
        self.robot_model = robot_model 
        self.config = config 

        init_channel(config=self.config)

        # Create simulator using factory 
        self.sim = SimulatorFactory.create_simulator(config=self.config, env_name=env_name, **kwargs)


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    # NOTE: NVIDIA team will override the interface to local if it is not specified 
    wbc_config["ENV_NAME"] = config.env_name 
    



if __name__ == "__main__":
    main()
