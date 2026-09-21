"""Run the Isaac Sim 6.0.1 to Unitree DDS adapter."""

import argparse
import sys

from gear_sonic.utils.isaacsim_simulator.configs import SimLoopConfig


def parse_args(argv=None) -> SimLoopConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd-path", required=True)
    parser.add_argument("--robot-path", default="/World/Robots/g1",
                        help="Robot subtree containing exactly one articulation root")
    parser.add_argument("--base-path", default="/World/Robots/g1/pelvis",
                        help="Pelvis rigid-body prim")
    parser.add_argument("--torso-path", default="/World/Robots/g1/torso_link",
                        help="Torso rigid-body prim")
    parser.add_argument("--physics-scene-path", default="/PhysicsScene",
                        help="PhysicsScene prim")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--interface", default="lo",
                        help="DDS interface; empty lets CycloneDDS choose")
    parser.add_argument("--physics-dt", type=float, default=0.002)
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--command-timeout", type=float, default=0.5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--inspect", action="store_true",
                        help="Validate scene and print joint map without DDS")
    hands = parser.add_mutually_exclusive_group()
    hands.add_argument("--with-hands", dest="with_hands", action="store_true",
                       help="Require integrated seven-DOF Dex3 hands")
    hands.add_argument("--without-hands", dest="with_hands", action="store_false",
                       help="Use only the 29 body joints, even if Dex3 joints are integrated")
    parser.set_defaults(with_hands=None)
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    return SimLoopConfig(**vars(parser.parse_args(argv)))


def main(argv=None) -> None:
    config = parse_args(argv)
    # SimulationApp parses sys.argv again and forwards unknown options to Kit.
    # Keep this adapter's CLI flags out of the Kit command line
    sys.argv = sys.argv[:1]
    from isaacsim import SimulationApp

    # DDS is closed explicitly below. Fast Kit shutdown avoids menu callbacks
    # racing a USD context that has already been destroyed during app.close()
    app = SimulationApp({"headless": config.headless, "fast_shutdown": True})
    simulator = None
    try:
        from gear_sonic.utils.isaacsim_simulator.simulator_factory import SimulatorFactory

        simulator = SimulatorFactory.create_simulator(config, app)
        SimulatorFactory.start_simulator(simulator)
    finally:
        if simulator is not None:
            simulator.close()
        app.close()


if __name__ == "__main__":
    main()
