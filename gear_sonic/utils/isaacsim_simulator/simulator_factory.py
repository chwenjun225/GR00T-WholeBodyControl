"""Factory for the Isaac Sim simulator and Unitree DDS setup."""

def init_channel(config):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    if config.interface:
        ChannelFactoryInitialize(config.domain_id, config.interface)
    else:
        ChannelFactoryInitialize(config.domain_id)


class SimulatorFactory:
    @staticmethod
    def create_simulator(config, simulation_app):
        from .base_sim import BaseSimulator
        return BaseSimulator(config, simulation_app)

    @staticmethod
    def start_simulator(simulator):
        try:
            simulator.start()
        except KeyboardInterrupt:
            print("Isaac Sim adapter interrupted by user.")
        finally:
            simulator.close()
