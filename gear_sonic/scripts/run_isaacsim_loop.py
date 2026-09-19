"""
Run the USD-to-Unitree DDS adapter using Isaac Sim's python.sh.
Example: ./python.sh -m gear_sonic.scripts.run_isaacsim_loop --help
"""
import tyro 

from gear_sonic.utils.isaacsim_simulator.configs import SimLoopConfig 

def main(config: SimLoopConfig):
    from isaacsim.simulation_app import SimulationApp
    simulation_app = SimulationApp({"headless": config.headless}) 
    simulator = None 
    try:
        from gear_sonic.utils.isaacsim_simulator.simulator_factory import SimulatorFactory 
        simulator = SimulatorFactory.create_simulator(config=config, simulation_app=simulation_app)
        if config.inspect: return 
        SimulatorFactory.start_simulator(simulator)
    finally:
        if simulator is not None: simulator.close()
        simulation_app.close() 

if __name__ == "__main__":
    config = tyro.cli(SimLoopConfig)
    main(config)