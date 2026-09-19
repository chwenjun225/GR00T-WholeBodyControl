"""Run the USD-to-Unitree DDS adapter using Isaac Sim's python.sh.

Example: ./python.sh -m gear_sonic.scripts.run_isaacsim_loop --help
"""
import argparse

from gear_sonic.utils.isaacsim_simulator.configs import SimLoopConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd-path", required=True)
    parser.add_argument("--robot-path", required=True, help="Robot subtree containing one articulation")
    parser.add_argument("--base-path", help="Optional explicit pelvis rigid-body prim")
    parser.add_argument("--torso-path", help="Optional explicit torso rigid-body prim")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--physics-dt", type=float, default=0.002)
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--command-timeout", type=float, default=0.5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="Print joint map and state; do not start DDS")
    parser.add_argument("--without-hands", dest="with_hands", action="store_false")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    return SimLoopConfig(**vars(parser.parse_args(argv)))


def main(argv=None):
    config = parse_args(argv)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": config.headless})
    try:
        # Kit-dependent modules must only be imported after app startup.
        from gear_sonic.utils.isaacsim_simulator.simulator_factory import SimulatorFactory
        simulator = SimulatorFactory.create_simulator(config, app)
        SimulatorFactory.start_simulator(simulator)
    finally:
        app.close()


if __name__ == "__main__":
    main()
