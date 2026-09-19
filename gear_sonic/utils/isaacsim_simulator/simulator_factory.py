"""Factory for creating and launching IsaacSim with Unitree SDK channel setup."""

def init_channel(config):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(config.domain_id, config.interface)


class SimulatorFactory:
    @staticmethod
    def create_simulator(config, simulation_app):
        from .base_sim import BaseSimulator
        return BaseSimulator(config, simulation_app)

    @staticmethod
    def start_simulator(simulator):
        try:
            simulator.start()
        finally:
            simulator.close()
